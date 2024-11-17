import csv
from datetime import datetime
import json
import os
import math
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')

def calculate_adjusted_score(score):
    try:
        score = int(score)
        result = math.floor((score - 100) / 30)
        logging.debug(f"Скорректированный балл: {result}")
        return result
    except (ValueError, TypeError) as e:
        logging.error(f"Ошибка при вычислении скорректированного балла: {e}, score={score}")
        return 0

last_game_number = 0
def process_game_data(data):
    player_game_data = {}
    game_number = 1

    for game in data:
        game_date_str = game.get("game_date")
        if not game_date_str:
            logging.warning(f"Пропуск игры без 'game_date'")
            continue
        try:
            game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
        except ValueError as e:
            logging.error(f"Недопустимый формат даты: {e}, game_date={game_date_str}")
            continue

        for round_data in game.get("rounds", []):
            for player_data in round_data.get("players", []):
                player_name = player_data.get("player_name")
                score_str = player_data.get("result")
                if player_name is None or score_str is None:
                    logging.warning(f"Неполные данные об игроке: {player_data}. Пропущено.")
                    continue

                try:
                    score = int(score_str)
                except ValueError as e:
                    logging.error(f"Неверный балл у {player_name}: {score_str}. Ошибка: {e}. Пропущено.")
                    continue

                if player_name not in player_game_data:
                    player_game_data[player_name] = {
                        "scores": [],
                        "game_numbers": [],
                        "adjusted_scores": [],
                        "dates": [],
                    }

                player_game_data[player_name]["scores"].append(score)
                player_game_data[player_name]["game_numbers"].append(game_number)
                player_game_data[player_name]["adjusted_scores"].append(calculate_adjusted_score(score))
                player_game_data[player_name]["dates"].append(game_date)
                game_number += 1

    return player_game_data


def save_and_create_bowling_report(json_filepath):
    output_dir = "bowling_reports"    
    logging.basicConfig(filename='bowling_report.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        with open(json_filepath, 'r') as f:
            game_data = json.load(f)
    except FileNotFoundError:
        logging.critical(f"Ошибка: файл JSON не найден по пути {json_filepath}")
        return
    except json.JSONDecodeError:
        logging.critical(f"Ошибка: Неверный формат JSON в {json_filepath}")
        return

    player_game_data = process_game_data(game_data)
    if not player_game_data:
        logging.warning("Не найдено достоверных данных об игроке. Создание CSV-файла пропущено.")
        return

    os.makedirs(output_dir, exist_ok=True)
    summary_filepath = os.path.join(output_dir, "bowling_summary.csv")
    details_filepath = os.path.join(output_dir, "bowling_details.csv")
    details_grid_filepath = os.path.join(output_dir, "bowling_details_grid.csv")

    try:
        for player_name, data in player_game_data.items():
            player_game_data[player_name]['adjusted_scores'] = [calculate_adjusted_score(score) for score in data['scores']]
            print(player_name)
            print(player_game_data[player_name])
                    
        # итоговый отчет
        with open(summary_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Игрок", "Сыграно", "Баллы", "Среднее", "Счет"])
            for player_name, data in player_game_data.items():
                total_score = sum(data["scores"])
                average_score = total_score / len(data["scores"]) if data["scores"] else 0
                total_adjusted_score = sum(data["adjusted_scores"])
                writer.writerow([player_name, len(data["scores"]), total_score, average_score, total_adjusted_score])

        # детальный отчет
        with open(details_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Игрок", "Номер игры", "Баллы", "Счет"])
            for player_name, data in player_game_data.items():
                for game_number, score, adjusted_score in zip(data["game_numbers"], data["scores"], data["adjusted_scores"]):
                    writer.writerow([player_name, game_number, score, adjusted_score])

        # детальный отчет в виде сетки
        with open(details_grid_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            #Header row
            header = ["Игрок"]
            max_games = max(len(data["scores"]) for data in player_game_data.values())
            for i in range(1, max_games + 1):
                header.extend([f"игра {i}", ""]) # result and score
            writer.writerow(header)
            #Data rows
            for player_name, data in player_game_data.items():
                row = [player_name]
                score_count = 0
                for i in range(max_games):
                    if score_count < len(data["scores"]):
                        row.extend([data["scores"][score_count], data["adjusted_scores"][score_count]])
                        score_count +=1
                    else:
                        row.extend(["", ""])
                writer.writerow(row)

    except (IOError, OSError) as e:
        logging.error(f"Ошибка при записи CSV-файлов: {e}")
    except Exception as e:
        logging.exception(f"При создании CSV-файла произошла непредвиденная ошибка: {e}")
        logging.info(f"Данные о боулинге сохранены в {summary_filepath} и {details_filepath}")

        
def create_and_save_graph(game_data):
    logging.info("Начинаю создание графика.")
    plt.figure(figsize=(10, 6))

    players = {}
    game_number_counter = 1

    for game in game_data:
        for round_data in game["rounds"]:
            for player_result in round_data["players"]:
                player_name = player_result["player_name"]
                score = player_result["result"]
                if player_name not in players:
                    players[player_name] = []
                players[player_name].append(score)
                game_number_counter += 1

    max_games = max(len(scores) for scores in players.values()) if players else 0
    for player_name, scores in players.items():
        plt.plot(range(1, len(scores) + 1), scores, label=player_name, marker='o')

    plt.xlabel("Номер игры")
    plt.xticks(list(range(1, max_games + 1)))  # Use max_games for x-axis ticks
    plt.ylabel("Баллы")
    plt.title("Результативность игроков")
    plt.legend()
    plt.grid(True)
    plt.savefig("bowling_reports\\player_results.png") # Use os.path.join for better path handling
    plt.close()
    logging.info("График сохранен в файл player_results.png.")

if __name__ == "__main__":
    json_filepath = "test1.json"
    save_and_create_bowling_report(json_filepath)
    create_and_save_graph(json.load(open(json_filepath, 'r')))
