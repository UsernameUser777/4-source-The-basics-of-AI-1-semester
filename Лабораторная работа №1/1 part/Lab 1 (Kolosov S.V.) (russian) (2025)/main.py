# Импортируем необходимые библиотеки
import numpy as np  # для работы с массивами и математическими операциями
import matplotlib.pyplot as plt  # для построения графиков
from tkinter import *  # для создания графического интерфейса
from tkinter import ttk, messagebox, filedialog, Toplevel, Text, StringVar, scrolledtext  # дополнительные виджеты tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # для встраивания графиков в tkinter
import logging  # для логирования действий
from logging.handlers import RotatingFileHandler  # для ограничения размера лог-файла
import datetime  # для работы с датами и временем

class FuzzyLogicApp:
    def __init__(self, root):
        # Инициализация главного окна приложения
        self.root = root
        self.root.title("Нечеткий вывод для управления вентилятором")

        # Настройка логирования
        self.logger = logging.getLogger('fuzzy_logic')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('fuzzy_logic.log', maxBytes=1000000, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Настройка стиля интерфейса
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TLabel', font=('Arial', 10))
        self.style.configure('TEntry', font=('Arial', 10))

        # Переменные для хранения значений температуры и номера варианта
        self.temperature = DoubleVar(value=22.0)
        self.variant_number = IntVar(value=1)

        # Виджеты для ввода температуры и варианта
        ttk.Label(root, text="Температура (°C):", font=('Arial', 10)).grid(column=0, row=0, padx=5, pady=5, sticky='w')
        ttk.Entry(root, textvariable=self.temperature, font=('Arial', 10)).grid(column=1, row=0, padx=5, pady=5)
        ttk.Button(root, text="Рассчитать", command=self.calculate).grid(column=2, row=0, padx=5, pady=5)

        ttk.Label(root, text="Номер варианта:", font=('Arial', 10)).grid(column=0, row=1, padx=5, pady=5, sticky='w')
        ttk.Entry(root, textvariable=self.variant_number, font=('Arial', 10)).grid(column=1, row=1, padx=5, pady=5)
        ttk.Button(root, text="Рассчитать 5 итераций", command=self.calculate_iterations).grid(column=2, row=1, padx=5, pady=5)

        # Кнопки настроек и действий
        ttk.Button(root, text="Настроить функции принадлежности", command=self.set_membership_parameters).grid(column=0, row=2, columnspan=3, padx=5, pady=5, sticky='ew')
        ttk.Button(root, text="Настроить правила", command=self.set_rules).grid(column=0, row=3, columnspan=3, padx=5, pady=5, sticky='ew')

        # Метки для вывода результатов
        self.result_label = ttk.Label(root, text="Скорость вентилятора: ", font=('Arial', 10))
        self.result_label.grid(column=0, row=4, columnspan=3, padx=5, pady=5, sticky='w')

        self.iteration_results_label = ttk.Label(root, text="Результаты итераций: ", font=('Arial', 10))
        self.iteration_results_label.grid(column=0, row=5, columnspan=3, padx=5, pady=5, sticky='w')

        # Кнопки сохранения и экспорта
        ttk.Button(root, text="Сохранить результаты", command=self.save_results).grid(column=0, row=6, columnspan=1, padx=5, pady=5, sticky='ew')
        ttk.Button(root, text="Экспорт графиков", command=self.export_plots).grid(column=1, row=6, columnspan=1, padx=5, pady=5, sticky='ew')
        ttk.Button(root, text="Справка", command=self.show_help).grid(column=2, row=6, columnspan=1, padx=5, pady=5, sticky='ew')
        ttk.Button(root, text="Сгенерировать отчет", command=self.generate_report).grid(column=0, row=7, columnspan=3, padx=5, pady=5, sticky='ew')

        # Журнал действий
        ttk.Label(root, text="Журнал действий:").grid(column=0, row=8, padx=5, pady=5, sticky='w')
        self.log_text = scrolledtext.ScrolledText(root, width=80, height=10, wrap=WORD)
        self.log_text.grid(column=0, row=9, columnspan=3, padx=5, pady=5, sticky='ew')
        ttk.Button(root, text="Очистить журнал", command=self.clear_log).grid(column=0, row=10, columnspan=3, padx=5, pady=5, sticky='ew')

        # Темы интерфейса
        self.theme_var = StringVar(value="light")
        ttk.Button(root, text="Сменить тему", command=self.toggle_theme).grid(column=0, row=11, columnspan=3, padx=5, pady=5, sticky='ew')

        # Создаем Canvas и Scrollbar для графиков
        self.canvas_frame = Frame(root)
        self.canvas_frame.grid(column=0, row=12, columnspan=3, padx=5, pady=5, sticky='nsew')

        self.canvas = Canvas(self.canvas_frame)
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.root.grid_rowconfigure(12, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.figures = []  # Список для хранения графиков

        # Параметры функций принадлежности
        self.low_temp_lower = DoubleVar(value=10.0)
        self.low_temp_upper = DoubleVar(value=30.0)
        self.medium_temp_lower = DoubleVar(value=10.0)
        self.medium_temp_upper = DoubleVar(value=50.0)
        self.high_temp_lower = DoubleVar(value=30.0)
        self.high_temp_upper = DoubleVar(value=50.0)

        # Правила нечеткой логики
        self.rules = [
            {"antecedent": "low_temp", "consequent": "low_speed"},
            {"antecedent": "medium_temp", "consequent": "medium_speed"},
            {"antecedent": "high_temp", "consequent": "high_speed"}
        ]

        # Обновление журнала действий
        self.update_log("Программа запущена")

    def update_log(self, message):
        # Обновление журнала действий
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {message}\n"
        self.log_text.insert(END, log_message)
        self.log_text.see(END)
        self.logger.info(message)

    def clear_log(self):
        # Очистка журнала действий
        self.log_text.delete(1.0, END)
        self.update_log("Журнал очищен")

    def calculate(self):
        # Расчет скорости вентилятора для заданной температуры
        try:
            temperature = self.temperature.get()
            if temperature < 0 or temperature > 60:
                raise ValueError("Температура должна быть в диапазоне от 0 до 60 °C")

            self.update_log(f"Рассчитываем скорость вентилятора для температуры: {temperature}°C")
            fan_speed, speed_range, low_speed_membership, medium_speed_membership, high_speed_membership, aggregated = calculate_fan_speed(temperature)
            self.result_label.config(text=f"Скорость вентилятора: {fan_speed:.2f} об/мин")

            # Графики для каждого этапа нечеткого вывода
            fig_fuzzification = plot_fuzzification(temperature)
            self.add_figure_to_scrollable_frame(fig_fuzzification)

            fig_aggregation = plot_aggregation(speed_range, low_speed_membership, medium_speed_membership, high_speed_membership)
            self.add_figure_to_scrollable_frame(fig_aggregation)

            fig_activation = plot_activation(speed_range, low_speed_membership, medium_speed_membership, high_speed_membership)
            self.add_figure_to_scrollable_frame(fig_activation)

            fig_accumulation = plot_accumulation(speed_range, aggregated)
            self.add_figure_to_scrollable_frame(fig_accumulation)

            fig_defuzzification = plot_defuzzification(speed_range, aggregated)
            self.add_figure_to_scrollable_frame(fig_defuzzification)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")
            self.update_log(f"Ошибка при расчете скорости вентилятора: {e}")

    def calculate_iterations(self):
        # Расчет пяти итераций с изменением температуры
        try:
            variant = self.variant_number.get()
            if variant < 1:
                raise ValueError("Номер варианта должен быть положительным числом")

            self.update_log(f"Рассчитываем 5 итераций для варианта: {variant}")
            temperatures, fan_speeds, areas, centroids = calculate_iterations(variant)

            results_text = "Результаты итераций:\n"
            for i, (t, v, area, centroid) in enumerate(zip(temperatures, fan_speeds, areas, centroids)):
                results_text += f"Итерация {i+1}: Температура = {t:.2f}°C, Скорость = {v:.2f} об/мин, Площадь = {area:.2f}, Центр тяжести = {centroid:.2f}\n"

            self.iteration_results_label.config(text=results_text)

            fig = plot_final_graph(temperatures, fan_speeds)
            self.add_figure_to_scrollable_frame(fig)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")
            self.update_log(f"Ошибка при расчете итераций: {e}")

    def add_figure_to_scrollable_frame(self, fig):
        # Добавление графика в прокручиваемый фрейм
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.figures.append(fig)

    def save_results(self):
        # Сохранение результатов в файл
        try:
            variant = self.variant_number.get()
            temperatures, fan_speeds, _, _ = calculate_iterations(variant)
            filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Текстовые файлы", "*.txt")])
            if filename:
                save_results(temperatures, fan_speeds, filename)
                messagebox.showinfo("Сохранение", "Результаты успешно сохранены!")
                self.update_log("Результаты успешно сохранены")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")
            self.update_log(f"Ошибка при сохранении результатов: {e}")

    def set_membership_parameters(self):
        # Настройка параметров функций принадлежности
        param_window = Toplevel(self.root)
        param_window.title("Настройка функций принадлежности")

        # Низкая температура
        ttk.Label(param_window, text="Низкая температура:").grid(column=0, row=0, padx=5, pady=5, sticky='w')
        ttk.Label(param_window, text="Нижняя граница:").grid(column=0, row=1, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.low_temp_lower).grid(column=1, row=1, padx=5, pady=5)
        ttk.Label(param_window, text="Верхняя граница:").grid(column=0, row=2, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.low_temp_upper).grid(column=1, row=2, padx=5, pady=5)

        # Средняя температура
        ttk.Label(param_window, text="Средняя температура:").grid(column=0, row=3, padx=5, pady=5, sticky='w')
        ttk.Label(param_window, text="Нижняя граница:").grid(column=0, row=4, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.medium_temp_lower).grid(column=1, row=4, padx=5, pady=5)
        ttk.Label(param_window, text="Верхняя граница:").grid(column=0, row=5, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.medium_temp_upper).grid(column=1, row=5, padx=5, pady=5)

        # Высокая температура
        ttk.Label(param_window, text="Высокая температура:").grid(column=0, row=6, padx=5, pady=5, sticky='w')
        ttk.Label(param_window, text="Нижняя граница:").grid(column=0, row=7, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.high_temp_lower).grid(column=1, row=7, padx=5, pady=5)
        ttk.Label(param_window, text="Верхняя граница:").grid(column=0, row=8, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.high_temp_upper).grid(column=1, row=8, padx=5, pady=5)

        ttk.Button(param_window, text="Применить", command=self.apply_membership_parameters).grid(column=0, row=9, columnspan=2, padx=5, pady=5)

    def apply_membership_parameters(self):
        # Применение новых параметров функций принадлежности
        messagebox.showinfo("Настройка", "Параметры функций принадлежности обновлены!")
        self.update_log("Параметры функций принадлежности обновлены")

    def export_plots(self):
        # Экспорт графиков в изображения
        if not self.figures:
            messagebox.showwarning("Экспорт", "Нет графиков для экспорта!")
            return

        directory = filedialog.askdirectory()
        if directory:
            for i, fig in enumerate(self.figures):
                fig.savefig(f"{directory}/plot_{i+1}.png", bbox_inches='tight', dpi=300)
            messagebox.showinfo("Экспорт", "Графики успешно экспортированы!")
            self.update_log("Графики успешно экспортированы")

    def toggle_theme(self):
        # Переключение темы интерфейса
        if self.theme_var.get() == "light":
            self.style.theme_use('alt')
            self.theme_var.set("dark")
            self.update_log("Темная тема применена")
        else:
            self.style.theme_use('clam')
            self.theme_var.set("light")
            self.update_log("Светлая тема применена")

    def show_help(self):
        # Отображение справки
        help_window = Toplevel(self.root)
        help_window.title("Справка")

        help_text = Text(help_window, wrap=WORD, font=('Arial', 10))
        help_text.insert(END, """
        Программа для управления вентилятором с использованием нечеткой логики.

        Постановка задачи:
        Задача кондиционера - поддерживать оптимальную температуру воздуха в комнате,
        охлаждая его, когда жарко, и нагревая, когда холодно.

        Правила корректировки скорости вращения вентилятора:
        1. Если температура низкая, то скорость вентилятора низкая.
        2. Если температура средняя, то скорость вентилятора средняя.
        3. Если температура высокая, то скорость вентилятора высокая.

        Использование:
        1. Введите температуру и нажмите "Рассчитать".
        2. Введите номер варианта и нажмите "Рассчитать 5 итераций".
        3. Используйте кнопку "Настроить функции принадлежности" для изменения параметров.
        4. Используйте кнопку "Сохранить результаты" для сохранения результатов в файл.
        5. Используйте кнопку "Экспорт графиков" для сохранения графиков в изображения.
        """)
        help_text.config(state=DISABLED)
        help_text.pack(padx=10, pady=10)
        self.update_log("Открыта справка")

    def set_rules(self):
        # Настройка правил нечеткой логики
        rules_window = Toplevel(self.root)
        rules_window.title("Настройка правил")

        self.rules_listbox = Listbox(rules_window, font=('Arial', 10))
        self.rules_listbox.pack(padx=10, pady=10, fill=BOTH, expand=True)

        for i, rule in enumerate(self.rules):
            self.rules_listbox.insert(END, f"Правило {i+1}: Если {rule['antecedent']}, то {rule['consequent']}")

        ttk.Button(rules_window, text="Добавить правило", command=self.add_rule).pack(padx=10, pady=5, fill=X)
        ttk.Button(rules_window, text="Удалить правило", command=self.delete_rule).pack(padx=10, pady=5, fill=X)
        self.update_log("Открыто окно настроек правил")

    def add_rule(self):
        # Добавление нового правила
        add_rule_window = Toplevel(self.root)
        add_rule_window.title("Добавить правило")

        self.antecedent_var = StringVar()
        self.consequent_var = StringVar()

        ttk.Label(add_rule_window, text="Антецедент:").grid(column=0, row=0, padx=5, pady=5, sticky='w')
        ttk.Entry(add_rule_window, textvariable=self.antecedent_var).grid(column=1, row=0, padx=5, pady=5)

        ttk.Label(add_rule_window, text="Консеквент:").grid(column=0, row=1, padx=5, pady=5, sticky='w')
        ttk.Entry(add_rule_window, textvariable=self.consequent_var).grid(column=1, row=1, padx=5, pady=5)

        ttk.Button(add_rule_window, text="Добавить", command=self.apply_add_rule).grid(column=0, row=2, columnspan=2, padx=5, pady=5)

    def apply_add_rule(self):
        # Применение нового правила
        antecedent = self.antecedent_var.get()
        consequent = self.consequent_var.get()

        if antecedent and consequent:
            self.rules.append({"antecedent": antecedent, "consequent": consequent})
            self.update_rules_list()
            messagebox.showinfo("Правила", "Новое правило добавлено!")
            self.update_log(f"Добавлено новое правило: Если {antecedent}, то {consequent}")

    def delete_rule(self):
        # Удаление правила
        selected_index = self.rules_listbox.curselection()
        if selected_index:
            del self.rules[selected_index[0]]
            self.update_rules_list()
            messagebox.showinfo("Правила", "Правило удалено!")
            self.update_log("Правило удалено")

    def update_rules_list(self):
        # Обновление списка правил
        self.rules_listbox.delete(0, END)
        for i, rule in enumerate(self.rules):
            self.rules_listbox.insert(END, f"Правило {i+1}: Если {rule['antecedent']}, то {rule['consequent']}")

    def generate_report(self):
        # Генерация отчета
        try:
            variant = self.variant_number.get()
            temperatures, fan_speeds, areas, centroids = calculate_iterations(variant)

            filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Текстовые файлы", "*.txt")])
            if filename:
                with open(filename, 'w') as file:
                    file.write("Отчет по лабораторной работе №1\n\n")
                    file.write("Постановка задачи:\n")
                    file.write("Задача кондиционера - поддерживать оптимальную температуру воздуха в комнате, охлаждая его, когда жарко, и нагревая, когда холодно.\n\n")

                    file.write("Правила корректировки скорости вращения вентилятора:\n")
                    for i, rule in enumerate(self.rules, start=1):
                        file.write(f"{i}. Если {rule['antecedent']}, то {rule['consequent']}\n")
                    file.write("\n")

                    file.write("Описание этапов нечеткого вывода:\n")
                    file.write("1. Фаззификация: вычисление значений функций принадлежности для нечетких множеств.\n")
                    file.write("2. Агрегирование подусловий: определение степени истинности составного высказывания.\n")
                    file.write("3. Активизация подзаключений: модификация нечетких множеств для правой части правил.\n")
                    file.write("4. Аккумуляция заключений: объединение модифицированных множеств.\n")
                    file.write("5. Дефаззификация: переход от нечеткого множества к скалярному значению.\n\n")

                    file.write("Результаты итераций:\n")
                    for i, (t, v, area, centroid) in enumerate(zip(temperatures, fan_speeds, areas, centroids), start=1):
                        file.write(f"Итерация {i}: Температура = {t:.2f}°C, Скорость = {v:.2f} об/мин, Площадь = {area:.2f}, Центр тяжести = {centroid:.2f}\n")

                messagebox.showinfo("Отчет", "Отчет успешно сгенерирован!")
                self.update_log("Отчет успешно сгенерирован")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")
            self.update_log(f"Ошибка при генерации отчета: {e}")

# Функция для построения графика фаззификации
def plot_fuzzification(temperature):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    temp_range = np.linspace(0, 60, 1000)
    ax.plot(temp_range, [low_temp_mf(t) for t in temp_range], label='Низкая', linewidth=2)
    ax.plot(temp_range, [medium_temp_mf(t) for t in temp_range], label='Средняя', linewidth=2)
    ax.plot(temp_range, [high_temp_mf(t) for t in temp_range], label='Высокая', linewidth=2)
    ax.axvline(x=temperature, color='r', linestyle='--', label=f'Текущая t={temperature}°C', linewidth=2)
    ax.set_title('Этап 1: Фаззификация', fontsize=12)
    ax.set_xlabel('Температура (°C)', fontsize=10)
    ax.set_ylabel('Степень принадлежности', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    return fig

# Функция для построения графика агрегирования
def plot_aggregation(speed_range, low_speed_membership, medium_speed_membership, high_speed_membership):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(speed_range, low_speed_membership, label='Низкая скорость', linewidth=2)
    ax.plot(speed_range, medium_speed_membership, label='Средняя скорость', linewidth=2)
    ax.plot(speed_range, high_speed_membership, label='Высокая скорость', linewidth=2)
    ax.set_title('Этап 2: Агрегирование подусловий', fontsize=12)
    ax.set_xlabel('Скорость (об/мин)', fontsize=10)
    ax.set_ylabel('Степень принадлежности', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    return fig

# Функция для построения графика активизации
def plot_activation(speed_range, low_speed_membership, medium_speed_membership, high_speed_membership):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(speed_range, low_speed_membership, label='Низкая скорость', linewidth=2)
    ax.plot(speed_range, medium_speed_membership, label='Средняя скорость', linewidth=2)
    ax.plot(speed_range, high_speed_membership, label='Высокая скорость', linewidth=2)
    ax.set_title('Этап 3: Активизация подзаключений', fontsize=12)
    ax.set_xlabel('Скорость (об/мин)', fontsize=10)
    ax.set_ylabel('Степень принадлежности', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    return fig

# Функция для построения графика аккумуляции
def plot_accumulation(speed_range, aggregated):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(speed_range, aggregated, label='Агрегированная', linestyle='--', linewidth=2)
    ax.set_title('Этап 4: Аккумуляция заключений', fontsize=12)
    ax.set_xlabel('Скорость (об/мин)', fontsize=10)
    ax.set_ylabel('Степень принадлежности', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    return fig

# Функция для построения графика дефаззификации
def plot_defuzzification(speed_range, aggregated):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(speed_range, aggregated, label='Агрегированная', linestyle='--', linewidth=2)
    ax.set_title('Этап 5: Дефаззификация', fontsize=12)
    ax.set_xlabel('Скорость (об/мин)', fontsize=10)
    ax.set_ylabel('Степень принадлежности', fontsize=10)
    ax.grid(True)

    # Найти центр тяжести
    centroid = defuzzify(speed_range, aggregated)
    ax.axvline(x=centroid, color='r', linestyle='--', label=f'Центр тяжести: {centroid:.2f} об/мин')

    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig

# Функции принадлежности для температуры
def low_temp_mf(t, low_lower=10.0, low_upper=30.0):
    if t <= low_lower:
        return 1.0
    elif low_lower < t <= low_upper:
        return (low_upper - t) / (low_upper - low_lower)
    else:
        return 0.0

def medium_temp_mf(t, medium_lower=10.0, medium_upper=50.0):
    if t <= medium_lower or t >= medium_upper:
        return 0.0
    elif medium_lower < t <= (medium_lower + medium_upper) / 2:
        return (t - medium_lower) / ((medium_lower + medium_upper) / 2 - medium_lower)
    elif (medium_lower + medium_upper) / 2 < t <= medium_upper:
        return (medium_upper - t) / (medium_upper - (medium_lower + medium_upper) / 2)
    else:
        return 1.0

def high_temp_mf(t, high_lower=30.0, high_upper=50.0):
    if t <= high_lower:
        return 0.0
    elif high_lower < t <= high_upper:
        return (t - high_lower) / (high_upper - high_lower)
    else:
        return 1.0

# Функции принадлежности для скорости вентилятора
def low_speed_mf(v):
    result = np.zeros_like(v, dtype=float)
    result[v <= 200] = 1.0
    mask = (v > 200) & (v <= 600)
    result[mask] = (600 - v[mask]) / 400
    return result

def medium_speed_mf(v):
    result = np.zeros_like(v, dtype=float)
    mask1 = (v > 200) & (v <= 600)
    result[mask1] = (v[mask1] - 200) / 400
    mask2 = (v > 600) & (v <= 800)
    result[mask2] = (800 - v[mask2]) / 200
    return result

def high_speed_mf(v):
    result = np.zeros_like(v, dtype=float)
    mask = (v > 600) & (v <= 800)
    result[mask] = (v[mask] - 600) / 200
    result[v > 800] = 1.0
    return result

# Дефаззификация методом центра тяжести
def defuzzify(speed_range, membership_values):
    numerator = np.trapz(speed_range * membership_values, speed_range)
    denominator = np.trapz(membership_values, speed_range)
    return numerator / denominator

# Основная функция для расчета скорости вентилятора
def calculate_fan_speed(temperature):
    low_temp = low_temp_mf(temperature)
    medium_temp = medium_temp_mf(temperature)
    high_temp = high_temp_mf(temperature)

    speed_range = np.linspace(0, 1000, 1000)

    low_speed_membership = np.fmin(low_speed_mf(speed_range), low_temp)
    medium_speed_membership = np.fmin(medium_speed_mf(speed_range), medium_temp)
    high_speed_membership = np.fmin(high_speed_mf(speed_range), high_temp)

    aggregated = np.fmax(low_speed_membership, np.fmax(medium_speed_membership, high_speed_membership))

    fan_speed = defuzzify(speed_range, aggregated)

    return fan_speed, speed_range, low_speed_membership, medium_speed_membership, high_speed_membership, aggregated

# Функция для построения графиков функций принадлежности
def plot_membership_functions(temperature, speed_range, low_speed_membership, medium_speed_membership, high_speed_membership, aggregated):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
    temp_range = np.linspace(0, 60, 1000)
    axs[0].plot(temp_range, [low_temp_mf(t) for t in temp_range], label='Низкая', linewidth=2)
    axs[0].plot(temp_range, [medium_temp_mf(t) for t in temp_range], label='Средняя', linewidth=2)
    axs[0].plot(temp_range, [high_temp_mf(t) for t in temp_range], label='Высокая', linewidth=2)
    axs[0].axvline(x=temperature, color='r', linestyle='--', label=f'Текущая t={temperature}°C', linewidth=2)
    axs[0].set_title('Функции принадлежности температуры', fontsize=12)
    axs[0].set_xlabel('Температура (°C)', fontsize=10)
    axs[0].set_ylabel('Степень принадлежности', fontsize=10)
    axs[0].legend(fontsize=10)
    axs[0].grid(True)
    axs[1].plot(speed_range, low_speed_membership, label='Низкая скорость', linewidth=2)
    axs[1].plot(speed_range, medium_speed_membership, label='Средняя скорость', linewidth=2)
    axs[1].plot(speed_range, high_speed_membership, label='Высокая скорость', linewidth=2)
    axs[1].plot(speed_range, aggregated, label='Агрегированная', linestyle='--', linewidth=2)
    axs[1].set_title('Функции принадлежности скорости вентилятора', fontsize=12)
    axs[1].set_xlabel('Скорость (об/мин)', fontsize=10)
    axs[1].set_ylabel('Степень принадлежности', fontsize=10)
    axs[1].legend(fontsize=10)
    axs[1].grid(True)
    plt.tight_layout()
    return fig

# Функция для построения итогового графика
def plot_final_graph(temperatures, fan_speeds):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(temperatures, fan_speeds, marker='o', linewidth=2)
    ax.set_title('Итоговый график изменения скорости вентилятора', fontsize=12)
    ax.set_xlabel('Температура (°C)', fontsize=10)
    ax.set_ylabel('Скорость вентилятора (об/мин)', fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    return fig

# Функция для расчета пяти итераций
def calculate_iterations(variant_number):
    temperatures = []
    fan_speeds = []
    areas = []
    centroids = []

    current_temp = 12 + 0.5 * variant_number
    temperatures.append(current_temp)

    for _ in range(5):
        fan_speed, speed_range, _, _, _, aggregated = calculate_fan_speed(current_temp)
        fan_speeds.append(fan_speed)

        # Расчет площади и центра тяжести
        area = np.trapz(aggregated, speed_range)
        centroid = defuzzify(speed_range, aggregated)
        areas.append(area)
        centroids.append(centroid)

        current_temp += 0.3
        temperatures.append(current_temp)

    return temperatures[:-1], fan_speeds, areas, centroids

# Функция для сохранения результатов
def save_results(temperatures, fan_speeds, filename="results.txt"):
    with open(filename, 'w') as file:
        file.write("Температура (°C)\tСкорость вентилятора (об/мин)\n")
        for t, v in zip(temperatures, fan_speeds):
            file.write(f"{t:.2f}\t{v:.2f}\n")

# Запуск приложения
root = Tk()
app = FuzzyLogicApp(root)
root.mainloop()
