import torch
from easyocr import Reader
import cv2
import os
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime

# Проверяем доступность устройства CUDA и выбираем первую видеокарту
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Загрузите модель YOLO
model = YOLO('bowlingTable_model.pt')  # Загрузка модели

# Используем EasyOCR для обработки текста
reader = Reader(['eu'], gpu=torch.cuda.is_available())  # Если CUDA доступна, используем GPU

# Создание директории для сохранения промежуточных изображений
output_dir = 'steps'
os.makedirs(output_dir, exist_ok=True)

# Путь к файлу с именами игроков
player_names_file = os.path.join('images', 'player_names.txt')

# Загрузка имен игроков из файла, если он существует
if os.path.exists(player_names_file):
    with open(player_names_file, 'r') as f:
        player_names = [line.strip() for line in f if line.strip()]
else:
    player_names = []

def correct_image(image, step_name):
    # 1. Коррекция "Света и тени"
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=-100)  # Света: -100
    image = cv2.addWeighted(image, 1.0, image, 0, -100)  # Тени: -100
    cv2.imwrite(os.path.join(output_dir, f'{step_name}_step1_light_shadow.jpg'), image)

    # 2. Увеличение чёткости
    blurred = cv2.GaussianBlur(image, (0, 0), 10)
    image = cv2.addWeighted(image, 2.0, blurred, -1.0, 0)  # Чёткость: 100, Радиус: 10
    cv2.imwrite(os.path.join(output_dir, f'{step_name}_step2_sharpen.jpg'), image)

    # 3. Перевод в чёрно-белый
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Возвращаем обратно в RGB для согласованности
    cv2.imwrite(os.path.join(output_dir, f'{step_name}_step3_bw.jpg'), image)

    # 4. Подсвеченные края (алгоритм Собеля)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)
    image = cv2.addWeighted(image, 1.0, cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB), 1.0, 0)
    cv2.imwrite(os.path.join(output_dir, f'{step_name}_step4_edges.jpg'), image)

    # 5. Коррекция яркости (Кривая яркости: (101, 134))
    lookup_table = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)]).astype('uint8')
    image = cv2.LUT(image, lookup_table)
    cv2.imwrite(os.path.join(output_dir, f'{step_name}_step5_brightness_curve.jpg'), image)

    return image

def perform_ocr(reader, image, original_image, step_name):
    # Пробуем распознать текст на обработанном изображении
    result = reader.readtext(image)
    if not result:
        # Если результат пуст, пробуем распознать текст на оригинальном изображении
        result = reader.readtext(original_image)
        if result:
            print(f"Обработанное изображение {step_name} не дало результатов, использован оригинал.")
            cv2.imwrite(os.path.join(output_dir, f'{step_name}_fallback_to_original.jpg'), original_image)
    return result

def extract_date_from_filename(filename):
    # Попробуем несколько форматов даты
    date_str = filename.split('_')[0]
    for date_format in ['%Y-%m-%d', '%Y-%m-%d_%H-%M-%S']:
        try:
            return datetime.strptime(date_str, date_format).date()
        except ValueError:
            continue
    # Если не удалось распарсить, вернуть None
    return None

def process_images():
    # Папка с изображениями
    images_folder = 'images'
    json_output = []

    # Группируем результаты по датам
    grouped_data = {}

    for image_name in os.listdir(images_folder):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_folder, image_name)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB

            # Сохраняем исходное изображение для проверки
            cv2.imwrite(os.path.join(output_dir, f'{image_name}_step1_original_image.jpg'), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

            # Используем YOLO для получения предсказаний на полном изображении
            results = model(image_rgb)  # Результаты предсказания

            # Инициализируем переменные для хранения областей game_number и player_scores
            game_number_bar_box = None
            player_scores_box = None
            game_number_boxes = []  # Список для хранения отдельных областей game_number
            player_score_boxes = []  # Список для хранения областей player_score

            # Обрабатываем результаты предсказания
            for result in results[0].boxes:
                xyxy = result.xyxy[0].cpu().numpy()  # Координаты [x1, y1, x2, y2]
                conf = result.conf[0].cpu().numpy()  # Достоверность
                cls = result.cls[0].cpu().numpy()  # Класс объекта

                if cls == 0:  # game_number_bar
                    game_number_bar_box = xyxy
                elif cls == 1:  # player_scores
                    player_scores_box = xyxy
                elif cls == 2:  # game_number
                    game_number_boxes.append(xyxy)
                elif cls == 3:  # player_score
                    player_score_boxes.append(xyxy)

            # Если не найдены game_number_bar или player_scores, используем отдельные области
            if (game_number_bar_box is None or game_number_bar_box.size == 0) and game_number_boxes:
                game_number_bar_box = game_number_boxes[0]

            if (player_scores_box is None or player_scores_box.size == 0) and player_score_boxes:
                x_min = min(box[0] for box in player_score_boxes)
                y_min = min(box[1] for box in player_score_boxes)
                x_max = max(box[2] for box in player_score_boxes)
                y_max = max(box[3] for box in player_score_boxes)
                player_scores_box = [x_min, y_min, x_max, y_max]

            # Сохраняем изображение с выделенными bounding box для проверки
            image_with_boxes = image_rgb.copy()
            if player_scores_box is not None:
                x1, y1, x2, y2 = map(int, player_scores_box)
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зелёный для player_scores
            if game_number_bar_box is not None:
                x1, y1, x2, y2 = map(int, game_number_bar_box)
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Красный для game_number_bar
            cv2.imwrite(os.path.join(output_dir, f'{image_name}_step2_image_with_boxes.jpg'), cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

            # Если нашли bounding box для game_number_bar, извлекаем, корректируем и распознаем текст
            game_number = None
            if game_number_bar_box is not None and len(game_number_bar_box) == 4:
                x1, y1, x2, y2 = map(int, game_number_bar_box)
                game_number_bar = image_rgb[y1:y2, x1:x2]
                corrected_game_number_bar = correct_image(game_number_bar, f'{image_name}_step3_game_number_bar')
                ocr_results_game_number = perform_ocr(reader, corrected_game_number_bar, game_number_bar, f'{image_name}_step4_ocr_game_number')
                if ocr_results_game_number:
                    game_number = ocr_results_game_number[0][1]  # Предполагаем, что результатом будет текст

            # Если нашли bounding box для player_scores, извлекаем, корректируем и распознаем текст
            player_scores = []
            if player_scores_box is not None and len(player_scores_box) == 4:
                x1, y1, x2, y2 = map(int, player_scores_box)
                player_scores_image = image_rgb[y1:y2, x1:x2]
                corrected_player_scores = correct_image(player_scores_image, f'{image_name}_step5_player_scores')
                ocr_results_player_scores = perform_ocr(reader, corrected_player_scores, player_scores_image, f'{image_name}_step6_ocr_player_scores')
                for bbox in ocr_results_player_scores:
                    if bbox[1] in player_names:
                        player_scores.append({'name': bbox[1], 'score': bbox[2]})  # Используем имя и оценку

            # Извлечение даты из имени файла
            game_date = extract_date_from_filename(image_name)

            # Если данные найдены, сохраняем их
            if game_number and player_scores:
                json_output.append({
                    'game_date': game_date.strftime('%Y-%m-%d') if game_date else None,
                    'game_number': game_number,
                    'player_scores': player_scores
                })

    # Сохраняем результат в JSON-файл
    with open('output.json', 'w') as json_file:
        json.dump(json_output, json_file, indent=4, ensure_ascii=False)

process_images()
