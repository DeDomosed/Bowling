import tkinter as tk
from tkinter import ttk, messagebox
import json
import openpyxl
import csv
import matplotlib.pyplot as plt
from openpyxl.styles import Font, Alignment, Side, Border, PatternFill, numbers


class ResultsPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Таблица для отображения результатов
        self.table = ttk.Treeview(self, columns=("game_number", "player_scores"), show="headings")
        self.table.heading("game_number", text="Номер игры")
        self.table.heading("player_scores", text="Очки игроков")
        self.table.pack(fill=tk.BOTH, expand=True)

        # Кнопки
        self.back_button = tk.Button(self, text="Вернуться", command=self.go_back)
        self.back_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.save_excel_button = tk.Button(self, text="Сохранить в Excel", command=self.save_to_excel)
        self.save_excel_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.plot_graph_button = tk.Button(self, text="Построить график", command=self.plot_graph)
        self.plot_graph_button.pack(side=tk.LEFT, padx=5, pady=10)

        # JSON данные
        self.json_data = []

    def display_results(self, json_file_path):
        """Загружает данные из JSON и отображает их в таблице."""
        print(f"Загружаем данные из файла: {json_file_path}")
        self.json_data = self.load_json(json_file_path)

        # Очищаем таблицу
        for row in self.table.get_children():
            self.table.delete(row)

        # Добавляем данные в таблицу
        for game in self.json_data:
            game_number = game.get("game_number")
            player_scores = ", ".join([f"{player['player_name']}: {player['result']}" for player in game["players"]])
            self.table.insert("", "end", values=(game_number, player_scores))

    def load_json(self, json_file_path):
        """Загружает данные из JSON-файла."""
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Данные из JSON: {data}")
            return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить JSON-файл: {e}")
            return []

    def save_to_csv(self):
        """Сохраняет данные из JSON в CSV-файл."""
        try:
            # Создаем CSV файл для записи
            with open("game_results.csv", "w", newline="", encoding="utf-8") as csvfile:
                csvwriter = csv.writer(csvfile)

                # Заголовок таблицы 1: Результаты игр
                csvwriter.writerow(["Номер игры", "Очки игроков"])

                # Записываем данные из JSON
                for game in self.json_data:
                    game_number = game.get("game_number")
                    player_scores = ", ".join(
                        [f"{player['player_name']}: {player['result']}" for player in game["players"]]
                    )
                    csvwriter.writerow([game_number, player_scores])

                # Пустая строка для разделения листов
                csvwriter.writerow([])

                # Заголовок таблицы 2: Общий счет и Среднее значение каждого игрока
                csvwriter.writerow(["Игрок", "Общий счет", "Среднее значение"])

                # Собираем данные по игрокам
                player_totals = {}  # Словарь для хранения общего счета каждого игрока
                player_counts = {}  # Словарь для подсчета количества игр каждого игрока

                for game in self.json_data:
                    for player in game["players"]:
                        player_name = player["player_name"]
                        score = player["result"]

                        if player_name not in player_totals:
                            player_totals[player_name] = 0
                            player_counts[player_name] = 0

                        player_totals[player_name] += score
                        player_counts[player_name] += 1

                # Записываем данные о каждом игроке
                for player_name in player_totals:
                    total_score = player_totals[player_name]
                    average_score = total_score / player_counts[player_name]
                    csvwriter.writerow([player_name, total_score, round(average_score, 2)])

            # Уведомляем об успешном сохранении
            messagebox.showinfo("Успех", "Данные успешно сохранены в CSV!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить в CSV: {e}")

    def plot_graph(self):
        """Строит график результатов и сохраняет его в .png."""
        try:
            # Убедимся, что данные из JSON загружены
            if not self.json_data:
                messagebox.showerror("Ошибка", "Нет данных для построения графика.")
                return

            # Собираем данные для построения графика
            players = {}
            for game in self.json_data:
                for player in game["players"]:
                    player_name = player["player_name"]
                    score = player["result"]
                    if player_name not in players:
                        players[player_name] = []
                    players[player_name].append(score)

            # Строим график
            plt.figure(figsize=(10, 6))
            for player_name, scores in players.items():
                plt.plot(range(1, len(scores) + 1), scores, label=player_name, marker='o')

            # Настройки графика
            plt.xlabel("Номер игры")
            plt.ylabel("Баллы")
            plt.title("Результативность игроков")
            plt.legend()
            plt.grid(True)

            # Сохраняем график в .png
            plt.savefig("player_results.png")
            plt.close()
            # Логика построения графика
            messagebox.showinfo("Успех", "График успешно построен и сохранен!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось построить график: {e}")

    def go_back(self):
        """Возвращает на главную страницу."""
        self.pack_forget()
        for widget in self.parent.winfo_children():
            widget.pack()
