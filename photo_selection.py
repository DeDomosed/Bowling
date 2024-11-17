import tkinter as tk
from tkinter import filedialog, ttk
from tkinterdnd2 import TkinterDnD
from PIL import Image, ImageTk
import os
import re
import json


class ImageApp(TkinterDnD.Tk):
    def __init__(self, min_width=350, min_height=350):
        super().__init__()
        self.title("Выбор файлов и Drag-and-Drop")
        self.geometry("800x600")

        # Список для хранения имен игроков
        self.player_names = []

        # Создаем таблицу для ввода имен игроков
        self.player_frame = tk.Frame(self)
        self.player_frame.pack(anchor="nw", padx=10, pady=10)

        tk.Label(self.player_frame, text="Имена игроков").pack(anchor="w")

        self.player_table = ttk.Treeview(self.player_frame, columns=("name"), show="headings", height=5)
        self.player_table.heading("name", text="Имя игрока")
        self.player_table.pack(fill=tk.X, expand=True)

        # Подключение событий для редактирования
        self.player_table.bind('<Double-1>', self.edit_player_name)

        # Кнопки для управления таблицей игроков
        self.add_name_button = tk.Button(self.player_frame, text="Добавить игрока", command=self.add_player_row)
        self.add_name_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.remove_name_button = tk.Button(self.player_frame, text="Удалить выбранного", command=self.remove_player_row)
        self.remove_name_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Остальной интерфейс
        self.choose_files_button = tk.Button(self, text="Выбрать файлы", command=self.choose_files)
        self.choose_files_button.pack(pady=10)

        self.drop_label = tk.Label(self, text="Или перетащите изображения сюда", bg="lightgray", width=50, height=5)
        self.drop_label.pack(pady=20)

        self.image_frame = tk.Frame(self, bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        self.drop_label.drop_target_register('DND_Files')
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)

        self.process_button = tk.Button(self, text="Обработать изображения", command=self.save_names_and_open_results)
        self.process_button.pack(pady=10)

        # Хранение изображений
        self.images = []

    def add_player_row(self):
        """Добавляет пустую строку для имени игрока."""
        new_index = len(self.player_names) + 1
        new_name = f"Игрок {new_index}"
        self.player_table.insert("", "end", values=(new_name,))
        self.player_names.append(new_name)

    def remove_player_row(self):
        """Удаляет выбранную строку."""
        selected_item = self.player_table.selection()
        if selected_item:
            for item in selected_item:
                self.player_table.delete(item)
                index = int(self.player_table.index(item))
                self.player_names.pop(index)

    def edit_player_name(self, event):
        """Позволяет редактировать имя игрока по двойному щелчку."""
        item_id = self.player_table.identify_row(event.y)
        column = self.player_table.identify_column(event.x)

        if item_id and column == '#1':
            bbox = self.player_table.bbox(item_id, column)
            entry = tk.Entry(self.player_frame)
            entry.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
            entry.insert(0, self.player_table.item(item_id, "values")[0])
            entry.focus()

            def save_edit(event=None):
                new_value = entry.get()
                self.player_table.item(item_id, values=(new_value,))
                entry.destroy()

            entry.bind("<Return>", save_edit)
            entry.bind("<FocusOut>", save_edit)

    def save_names_to_json(self):
        """Сохраняет введенные имена игроков в JSON-файл."""
        try:
            players_data = [{"player_name": self.player_table.item(row_id, "values")[0]}
                            for row_id in self.player_table.get_children()]

            with open("players.json", "w", encoding="utf-8") as f:
                json.dump(players_data, f, ensure_ascii=False, indent=4)

            print("Имена игроков успешно сохранены в 'players.json'.")
        except Exception as e:
            print(f"Ошибка при сохранении имен игроков: {e}")

    def save_names_and_open_results(self):
        """Сохраняет имена игроков и открывает страницу результатов."""
        self.save_names_to_json()
        print("Открытие страницы результатов...")

    def choose_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Выберите изображения",
            filetypes=(("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Все файлы", "*.*"))
        )
        if file_paths:
            for file_path in file_paths:
                self.display_image(file_path)

    def on_drop(self, event):
        file_paths = self.extract_file_paths(event.data)
        if not file_paths:
            self.drop_label.config(text="Ошибка: не удалось извлечь файлы из перетаскивания")
            return

        for file_path in file_paths:
            file_path = file_path.strip('{}')
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.display_image(file_path)
            else:
                self.drop_label.config(text=f"Файл '{file_path}' не является изображением")

    def extract_file_paths(self, data):
        file_paths = re.findall(r'\{[^}]+\}|[^\s\{]+', data)
        if not file_paths:
            file_paths = data.split()
        return file_paths

    def display_image(self, file_path):
        try:
            if not os.path.exists(file_path):
                self.drop_label.config(text=f"Ошибка: файл '{file_path}' не существует")
                return

            self.images.append(file_path)
            image = Image.open(file_path)
            image_width = 200
            image_height = 150
            image.thumbnail((image_width, image_height))
            photo = ImageTk.PhotoImage(image)

            image_label = tk.Label(self.image_frame, image=photo)
            image_label.photo = photo

            i = len(self.images) - 1
            columns_count = self.image_frame.winfo_width() // image_width
            image_label.grid(row=i // columns_count, column=i % columns_count)

            image_label.bind("<Button-1>", lambda event, path=file_path: self.open_large_image(path))

        except Exception as e:
            self.drop_label.config(text=f"Ошибка при загрузке изображения: {str(e)}")

    def open_large_image(self, file_path):
        try:
            image = Image.open(file_path)
            max_width = 800
            max_height = 600
            img_width, img_height = image.size
            aspect_ratio = img_width / img_height

            if img_width > max_width or img_height > max_height:
                if aspect_ratio > 1:
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)

                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            top = tk.Toplevel(self)
            top.title("Просмотр изображения")

            photo = ImageTk.PhotoImage(image)
            image_label = tk.Label(top, image=photo)
            image_label.photo = photo
            image_label.pack(padx=20, pady=20)

            top.geometry(f"{image.width + 40}x{image.height + 40}")

            close_button = tk.Button(top, text="Закрыть", command=top.destroy)
            close_button.pack(pady=10)

        except Exception as e:
            print(f"Не удалось открыть изображение: {e}")


if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()
