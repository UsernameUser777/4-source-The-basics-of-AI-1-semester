import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
import json
import os
import csv
import itertools
import threading
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Попытка импорта из time_series_utils.py
try:
    import time_series_utils
    HAS_TSU = True
    logger.info("time_series_utils.py импортирован.")
except ImportError:
    HAS_TSU = False
    logger.info("time_series_utils.py не найден, используется собственная реализация.")


class TimeSeriesDatasetProcessor:
    """
    Класс для обработки временного ряда: загрузка, нормализация, создание окон, разделение выборки.
    """
    def __init__(self, dataset_path, depth, scaler_range=(-1, 1), use_tsu=HAS_TSU):
        """
        Инициализация процессора данных.

        Args:
            dataset_path (str): Путь к CSV-файлу датасета.
            depth (int): Глубина погружения (n) для метода окон.
            scaler_range (tuple): Диапазон для нормализации (min, max).
            use_tsu (bool): Использовать ли time_series_utils.py.
        """
        logger.info(f"Инициализация процессора данных для {dataset_path} (use_tsu={use_tsu})")
        self.dataset_path = dataset_path
        self.depth = depth
        self.scaler_range = scaler_range
        self.use_tsu = use_tsu
        self.scaler = MinMaxScaler(feature_range=scaler_range)
        self.data = None  # Исходный временной ряд
        self.normalized_data = None  # Нормализованный ряд
        self.X = None  # Входные признаки (окна)
        self.y = None  # Целевые значения (следующее значение после окна)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_class = None # Целевые значения для классификации
        self.y_test_class = None
        self.scaler_used = scaler_range # Для информации

        if self.use_tsu:
            # Используем time_series_utils
            self.ts_dataset = time_series_utils.TimeSeriesDataset(
                filepath=dataset_path,
                depth=depth,
                horizon=1,
                feature_range=scaler_range
            )
            self.ts_dataset.load_and_process()
            self.X, self.y = self.ts_dataset.get_features_and_targets()
            self.scaler = self.ts_dataset.scaler # Используем scaler из TSU

    def load_and_preprocess(self):
        """
        Загружает данные и выполняет полную предварительную обработку:
        загрузка, нормализация, создание окон, разделение на обучающую и тестовую выборки.
        """
        logger.info(f"Загрузка данных из {self.dataset_path}")
        if self.use_tsu:
            logger.info("Использование time_series_utils для предобработки.")
            # Уже выполнено в __init__ если use_tsu=True
            self.X, self.y = self.ts_dataset.get_features_and_targets()
            self.X_train, self.X_test, self.y_train, self.y_test = self.ts_dataset.split_data(train_ratio=0.7)
            self.data = self.ts_dataset.raw_data.flatten() # Извлекаем исходные данные из TSU
        else:
            logger.info("Использование собственной реализации для предобработки.")
            # Пытаемся загрузить исходный временной ряд из all_datasets_combined.csv
            all_data_path = self.dataset_path.replace("dataset_15_Sensor_Events.csv", "all_datasets_combined.csv")
            if os.path.exists(all_data_path):
                logger.info(f"Загрузка исходного ряда из {all_data_path}")
                full_df = pd.read_csv(all_data_path)
                # Извлекаем ID датасета из названия файла
                try:
                    dataset_id = int(self.dataset_path.split('_')[1])
                except (IndexError, ValueError):
                    dataset_id = 15 # По умолчанию
                dataset_df = full_df[full_df['Dataset_ID'] == dataset_id].copy()
                self.data = dataset_df['Value'].values.astype(float)
            else:
                # Если all_datasets_combined.csv недоступен, восстанавливаем ряд из файла с окнами
                logger.info("Файл all_datasets_combined.csv не найден, восстановление ряда из CSV с окнами")
                df_windows = pd.read_csv(self.dataset_path)
                # Восстановление: первые depth значений из Input_1 ... Input_depth + все Output
                first_inputs = df_windows.iloc[0, :self.depth].values
                outputs = df_windows['Output'].values
                # Конкатенируем: [Input_1, Input_2, ..., Input_depth, Output_1, Output_2, ...]
                self.data = np.concatenate([first_inputs, outputs])
                logger.info(f"Ряд восстановлен. Длина: {len(self.data)}")

            logger.info(f"Загружено {len(self.data)} точек данных.")

            # Нормализация
            self.normalized_data = self.scaler.fit_transform(self.data.reshape(-1, 1)).flatten()
            logger.info("Данные нормализованы.")

            # Создание окон
            self.X, self.y = self.create_windows(self.normalized_data, self.depth, horizon=1)
            logger.info(f"Создано {len(self.X)} обучающих примеров методом скользящего окна.")

            # Разделение на обучающую и тестовую выборки (70/30)
            split_idx = int(len(self.X) * 0.7)
            self.X_train = self.X[:split_idx]
            self.X_test = self.X[split_idx:]
            self.y_train = self.y[:split_idx]
            self.y_test = self.y[split_idx:]
            logger.info(f"Разделение: {len(self.X_train)} тренировка, {len(self.X_test)} тест.")

        # Создание классов для задачи классификации
        # Используем пороги из datasets_description.pdf для датасета 15
        # 1: < 25, 2: 25-50, 3: > 50
        # Денормализуем y_train и y_test для определения классов
        y_train_denorm = self.inverse_transform(self.y_train)
        y_test_denorm = self.inverse_transform(self.y_test)
        self.y_train_class = self.value_to_class(y_train_denorm)
        self.y_test_class = self.value_to_class(y_test_denorm)

    def value_to_class(self, values):
        """Преобразует значения в классы для датасета 15."""
        values = np.array(values).flatten()
        classes = np.zeros_like(values, dtype=int)
        classes[values < 25] = 1
        classes[(values >= 25) & (values <= 50)] = 2
        classes[values > 50] = 3
        return classes

    def create_windows(self, data, depth, horizon):
        """
        Создает окна для обучения из одномерного временного ряда.
        Используется, если use_tsu=False.
        """
        X, y = [], []
        for i in range(len(data) - depth - horizon + 1):
            X.append(data[i:i + depth])
            y.append(data[i + depth:i + depth + horizon][0]) # Только первое значение после окна
        return np.array(X), np.array(y)

    def inverse_transform(self, value):
        """
        Восстанавливает значение из нормализованного масштаба в исходный.
        """
        if self.use_tsu:
            return self.ts_dataset.inverse_normalize(value)
        else:
            return self.scaler.inverse_transform(np.array(value).reshape(-1, 1))

class MLPForecaster:
    """
    Класс для построения, обучения и оценки многослойного перцептрона.
    """
    def __init__(self, input_size, hidden_size, output_size, activation_func, learning_rate=0.05, optimizer='SGD', use_dropout=False, dropout_rate=0.0, use_batch_norm=False, task='regression'):
        """
        Инициализация модели.

        Args:
            input_size (int): Размерность входного слоя (глубина погружения).
            hidden_size (int): Размерность скрытого слоя.
            output_size (int): Размерность выходного слоя (обычно 1).
            activation_func (str): Функция активации ('tanh' или 'sigmoid').
            learning_rate (float): Начальный шаг обучения.
            optimizer (str): Оптимизатор ('SGD', 'Adam', 'RMSprop').
            use_dropout (bool): Использовать ли Dropout.
            dropout_rate (float): Степень Dropout.
            use_batch_norm (bool): Использовать ли BatchNormalization.
            task (str): 'regression' или 'classification'.
        """
        logger.info(f"Инициализация MLPForecaster: {input_size} -> {hidden_size} ({activation_func}) -> {output_size} (task={task})")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_func = activation_func
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.task = task
        self.model = None
        self.history = None

    def build_model(self):
        """
        Создает архитектуру нейронной сети.
        Архитектура: 2 слоя (входной неявный, скрытый, выходной).
        """
        logger.info(f"Создание модели: {self.input_size} -> {self.hidden_size} ({self.activation_func}) -> {self.output_size} (task={self.task})")
        self.model = keras.Sequential([
            layers.Dense(self.hidden_size, activation=self.activation_func, input_shape=(self.input_size,)),
        ])

        if self.use_batch_norm:
            self.model.add(layers.BatchNormalization())
        if self.use_dropout:
            self.model.add(layers.Dropout(self.dropout_rate))

        if self.task == 'regression':
            self.model.add(layers.Dense(self.output_size, activation='linear'))
            loss = 'mean_squared_error'
        elif self.task == 'classification':
            self.model.add(layers.Dense(self.output_size, activation='softmax')) # 3 класса
            loss = 'sparse_categorical_crossentropy'

        if self.optimizer_name == 'SGD':
            optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'RMSprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate) # По умолчанию

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['mae' if self.task == 'regression' else 'accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10000, patience=50, verbose=1, progress_callback=None):
        """
        Обучает модель.

        Args:
            X_train, y_train: Обучающая выборка.
            X_val, y_val: Валидационная выборка (используется для ранней остановки).
            epochs (int): Максимальное количество эпох.
            patience (int): Пациенция для ранней остановки.
            verbose (int): 0 - без логов, 1 - с прогресс-баром, 2 - для каждой эпохи.
            progress_callback (func): Функция для обновления прогресс-бара.

        Returns:
            history: История обучения.
        """
        if self.model is None:
            raise ValueError("Модель не создана. Вызовите build_model() сначала.")

        logger.info(f"Начало обучения модели. Epochs: {epochs}, Patience: {patience}")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1
            )
        ]
        if progress_callback:
            callbacks.append(ProgressCallback(epochs, progress_callback))

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val), # Используем тест как валидацию для ранней остановки
            epochs=epochs,
            batch_size=1, # Стохастическое обучение (r=0.05)
            callbacks=callbacks,
            verbose=verbose
        )
        logger.info("Обучение завершено.")
        return self.history

    def predict(self, X):
        """
        Выполняет предсказание.
        """
        if self.model is None:
            raise ValueError("Модель не обучена.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, y_test_class=None):
        """
        Оценивает модель на тестовой выборке.

        Args:
            X_test, y_test: Тестовая выборка (для регрессии).
            y_test_class: Тестовая выборка (для классификации).

        Returns:
            dict: Словарь с метриками и предсказаниями.
        """
        if self.model is None:
            raise ValueError("Модель не обучена.")
        y_pred = self.predict(X_test)

        results = {}
        if self.task == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            logger.info(f"Тестовая MSE (норм.): {mse:.6f}, MAE (норм.): {mae:.6f}, RMSE (норм.): {rmse:.6f}")
            results.update({"MSE": mse, "MAE": mae, "RMSE": rmse, "y_pred": y_pred})
        elif self.task == 'classification':
            y_pred_class = np.argmax(y_pred, axis=1) # Получаем предсказанные классы
            accuracy = accuracy_score(y_test_class, y_pred_class)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test_class, y_pred_class, average='macro', zero_division=0)
            conf_matrix = confusion_matrix(y_test_class, y_pred_class)
            logger.info(f"Тестовая Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}")
            results.update({"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "Confusion_Matrix": conf_matrix, "y_pred_class": y_pred_class})

        return results

class ProgressCallback(keras.callbacks.Callback):
    """Кастомный колбэк для обновления прогресс-бара."""
    def __init__(self, total_epochs, progress_callback):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_callback = progress_callback

    def on_epoch_end(self, epoch, logs=None):
        progress = int(100 * (epoch + 1) / self.total_epochs)
        self.progress_callback(progress)

class FuzzyLogicApp:
    """
    Класс для графического интерфейса tkinter.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Прогнозирование временных рядов - MLP (Вариант 15)")

        # --- Загрузка параметров датасетов ---
        self.datasets_params_df = self.load_datasets_params()

        # --- Переменные для хранения параметров ---
        self.dataset_path_var = tk.StringVar(value="dataset_15_Sensor_Events.csv")
        self.dataset_list = self.find_dataset_files() # Список файлов датасетов
        self.dataset_combobox_var = tk.StringVar(value="dataset_15_Sensor_Events.csv")
        self.depth_var = tk.IntVar(value=6) # Для датасета 15 из datasets_parameters.csv
        self.hidden_neurons_var = tk.IntVar(value=12) # Рекомендуемое начальное значение для датасета 15
        self.activation_var = tk.StringVar(value="tanh")
        self.learning_rate_var = tk.DoubleVar(value=0.05) # Из задания ЛР
        self.max_error_var = tk.DoubleVar(value=0.005) # Из задания ЛР

        # --- Новые переменные ---
        self.optimizer_var = tk.StringVar(value="SGD")
        self.use_dropout_var = tk.BooleanVar(value=False)
        self.dropout_rate_var = tk.DoubleVar(value=0.1)
        self.use_batch_norm_var = tk.BooleanVar(value=False)
        self.task_var = tk.StringVar(value="regression") # 'regression' или 'classification'
        self.use_tsu_var = tk.BooleanVar(value=HAS_TSU) # Использовать ли time_series_utils

        # --- Переменные для серии экспериментов ---
        self.hidden_neurons_list_var = tk.StringVar(value="8, 12, 18")
        self.activation_list_var = tk.StringVar(value="tanh, sigmoid")
        self.depth_list_var = tk.StringVar(value="4, 6") # Новый список для глубины
        self.dataset_list_var = tk.StringVar(value="dataset_15_Sensor_Events.csv") # Новый список для датасетов

        # --- Переменные для хранения результатов ---
        self.current_results = {}
        self.current_history = None
        self.data_processor = None
        self.forecaster = None

        # --- Стиль ---
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # --- Создание меню ---
        self.setup_menu()

        # --- Создание вкладок ---
        self.setup_ui()

    def load_datasets_params(self):
        """Загружает datasets_parameters.csv в DataFrame."""
        try:
            df = pd.read_csv("datasets_parameters.csv")
            logger.info("datasets_parameters.csv загружен.")
            return df
        except FileNotFoundError:
            logger.warning("datasets_parameters.csv не найден. Используются значения по умолчанию.")
            return None

    def find_dataset_files(self):
        """Находит все файлы dataset_*.csv в текущей директории."""
        import glob
        files = glob.glob("dataset_*.csv")
        files.sort()
        return files

    def setup_menu(self):
        """Создает меню."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Меню "Файл"
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Открыть датасет...", command=self.browse_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)

        # Меню "Настройки"
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        settings_menu.add_command(label="Сохранить конфигурацию...", command=self.save_config)
        settings_menu.add_command(label="Загрузить конфигурацию...", command=self.load_config)

        # Меню "Справка"
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_about)

    def setup_ui(self):
        """
        Создает элементы интерфейса с использованием вкладок.
        """
        # Создаем NoteBook (вкладки)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Вкладка 1: Параметры и Управление ---
        tab_params = ttk.Frame(notebook)
        notebook.add(tab_params, text="Параметры и Управление")

        # Размещаем параметры и кнопки управления на этой вкладке
        self.setup_parameters_frame(tab_params)
        self.setup_control_buttons_frame(tab_params)

        # --- Вкладка 2: Журнал действий ---
        tab_log = ttk.Frame(notebook)
        notebook.add(tab_log, text="Журнал действий")
        self.setup_log_frame(tab_log)

        # --- Вкладка 3: Результаты и Таблица ---
        tab_results = ttk.Frame(notebook)
        notebook.add(tab_results, text="Результаты и Таблица")
        self.setup_results_table_frame(tab_results)

        # --- Вкладка 4: Графики ---
        tab_plots = ttk.Frame(notebook)
        notebook.add(tab_plots, text="Графики")
        self.setup_plots_frame(tab_plots)

        # --- Вкладка 5: Описание датасета и Листинг ---
        tab_info = ttk.Frame(notebook)
        notebook.add(tab_info, text="Описание и Листинг")
        self.setup_info_frame(tab_info)

        # --- Вкладка 6: Прогноз ---
        tab_forecast = ttk.Frame(notebook)
        notebook.add(tab_forecast, text="Прогноз")
        self.setup_forecast_frame(tab_forecast)

        # Настройка размеров
        self.root.geometry("1200x800") # Начальный размер

    def setup_parameters_frame(self, parent):
        """Создает фрейм с параметрами модели."""
        params_frame = ttk.LabelFrame(parent, text="Параметры модели")
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        # --- Основные параметры ---
        ttk.Label(params_frame, text="Выбор датасета:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        dataset_combo = ttk.Combobox(params_frame, textvariable=self.dataset_combobox_var, values=self.dataset_list, state="readonly", width=30)
        dataset_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        dataset_combo.bind("<<ComboboxSelected>>", self.on_dataset_combo_change) # Обновляем путь при выборе

        ttk.Label(params_frame, text="Глубина погружения (n):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(params_frame, from_=1, to=20, textvariable=self.depth_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(params_frame, text="Число нейронов скрытого слоя:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(params_frame, from_=1, to=50, textvariable=self.hidden_neurons_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(params_frame, text="Функция активации:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        activation_combo = ttk.Combobox(params_frame, textvariable=self.activation_var, values=["tanh", "sigmoid"], state="readonly", width=8)
        activation_combo.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(params_frame, text="Начальный шаг обучения (r):").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(params_frame, textvariable=self.learning_rate_var, width=10).grid(row=4, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(params_frame, text="Макс. допустимая ошибка (ε):").grid(row=5, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(params_frame, textvariable=self.max_error_var, width=10).grid(row=5, column=1, sticky="w", padx=5, pady=2)

        # --- Новые параметры ---
        ttk.Label(params_frame, text="Оптимизатор:").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        optimizer_combo = ttk.Combobox(params_frame, textvariable=self.optimizer_var, values=["SGD", "Adam", "RMSprop"], state="readonly", width=8)
        optimizer_combo.grid(row=6, column=1, sticky="w", padx=5, pady=2)

        ttk.Checkbutton(params_frame, text="Использовать Dropout", variable=self.use_dropout_var).grid(row=7, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(params_frame, textvariable=self.dropout_rate_var, width=10).grid(row=7, column=1, sticky="w", padx=5, pady=2)

        ttk.Checkbutton(params_frame, text="Использовать BatchNormalization", variable=self.use_batch_norm_var).grid(row=8, column=0, sticky="w", padx=5, pady=2)

        ttk.Label(params_frame, text="Задача:").grid(row=9, column=0, sticky="w", padx=5, pady=2)
        task_combo = ttk.Combobox(params_frame, textvariable=self.task_var, values=["regression", "classification"], state="readonly", width=8)
        task_combo.grid(row=9, column=1, sticky="w", padx=5, pady=2)

        ttk.Checkbutton(params_frame, text="Использовать time_series_utils", variable=self.use_tsu_var).grid(row=10, column=0, sticky="w", padx=5, pady=2)

        # --- Параметры для серии экспериментов ---
        exp_frame = ttk.LabelFrame(parent, text="Параметры для серии экспериментов (через запятую)")
        exp_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(exp_frame, text="Список числа нейронов:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(exp_frame, textvariable=self.hidden_neurons_list_var, width=20).grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(exp_frame, text="Список функций активации:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(exp_frame, textvariable=self.activation_list_var, width=20).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(exp_frame, text="Список глубин (depth):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(exp_frame, textvariable=self.depth_list_var, width=20).grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(exp_frame, text="Список датасетов:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(exp_frame, textvariable=self.dataset_list_var, width=20).grid(row=3, column=1, sticky="w", padx=5, pady=2)

    def setup_control_buttons_frame(self, parent):
        """Создает фрейм с кнопками управления."""
        # Создаем Canvas с горизонтальной прокруткой для кнопок
        canvas_container = tk.Frame(parent)
        canvas_container.pack(fill=tk.X, padx=10, pady=5)

        button_canvas = tk.Canvas(canvas_container)
        button_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        scrollbar = ttk.Scrollbar(canvas_container, orient="horizontal", command=button_canvas.xview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.X)

        button_canvas.configure(xscrollcommand=scrollbar.set)

        button_inner_frame = tk.Frame(button_canvas)
        button_canvas.create_window((0, 0), window=button_inner_frame, anchor="nw")

        # Настройка прокрутки
        def on_configure(event):
            button_canvas.configure(scrollregion=button_canvas.bbox("all"))

        button_inner_frame.bind("<Configure>", on_configure)

        # Добавляем кнопки в inner_frame
        buttons = [
            ("Загрузить и обработать данные", self.load_and_process_data),
            ("Обучить модель", self.train_model),
            ("Оценить модель", self.evaluate_model),
            ("Показать графики", self.show_plots_in_new_window),
            ("Сохранить результаты", self.save_results),
            ("Экспорт таблицы", self.export_table),
            ("Показать первые 10 окон", self.show_training_samples),
            ("Копировать листинг", self.copy_code),
            ("Сохранить модель", self.save_trained_model),
            ("Запустить серию экспериментов", self.run_experiment_series),
            ("Сохранить историю обучения", self.save_training_history),
            ("Загрузить историю обучения", self.load_training_history),
            ("Прогноз на следующий шаг", self.predict_next_step)
        ]

        for text, command in buttons:
            btn = ttk.Button(button_inner_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5, pady=5)

    def setup_log_frame(self, parent):
        """Создает фрейм для журнала действий."""
        log_frame = ttk.LabelFrame(parent, text="Журнал действий")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Прогресс-бар
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(log_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

    def setup_results_table_frame(self, parent):
        """Создает фрейм для результатов и таблицы."""
        results_frame = ttk.LabelFrame(parent, text="Результаты")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, side=tk.TOP)

        self.results_text = scrolledtext.ScrolledText(results_frame, state='disabled')
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        table_frame = ttk.LabelFrame(parent, text="Таблица сравнения результатов")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, side=tk.BOTTOM)

        # Создаем Treeview для таблицы
        # Обновленные столбцы
        columns = ("Dataset", "Task", "Activation", "Optimizer", "Hidden Neurons", "Depth", "Uses Dropout", "Dropout Rate", "Uses BatchNorm", "MSE_norm", "MAE_norm", "RMSE_norm", "MSE_actual", "MAE_actual", "RMSE_actual", "Accuracy", "F1-Score")
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        self.results_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5, side=tk.LEFT)

        # Настройка заголовков
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=80, anchor='center') # Уменьшаем ширину

        # Добавляем скроллбар для Treeview
        tree_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.configure(yscrollcommand=tree_scrollbar.set)

        # Добавим кнопку для добавления текущих результатов в таблицу
        add_button = ttk.Button(table_frame, text="Добавить текущие результаты в таблицу", command=self.add_to_table)
        add_button.pack(pady=5)

    def setup_plots_frame(self, parent):
        """Создает фрейм для графиков."""
        plot_frame = ttk.LabelFrame(parent, text="Графики")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Создаем Canvas с Scrollbar
        canvas_container = tk.Frame(plot_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas для отрисовки графиков
        self.plot_canvas = tk.Canvas(canvas_container)
        self.plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.plot_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Связываем Canvas и Scrollbar
        self.plot_canvas.configure(yscrollcommand=scrollbar.set)

        # Frame внутри Canvas, где будут размещены графики
        self.plot_inner_frame = tk.Frame(self.plot_canvas)
        self.plot_canvas.create_window((0, 0), window=self.plot_inner_frame, anchor="nw")

        # Настройка прокрутки
        def on_configure(event):
            self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))

        self.plot_inner_frame.bind("<Configure>", on_configure)

    def setup_info_frame(self, parent):
        """Создает фрейм для описания датасета и листинга программы."""
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Описание датасета
        desc_frame = ttk.LabelFrame(info_frame, text="Описание датасета 15")
        desc_frame.pack(fill=tk.X, padx=5, pady=5)

        desc_text = (
            "ДАТАСЕТ 15: СРАБАТЫВАНИЯ ДАТЧИКОВ БЕЗОПАСНОСТИ\n"
            "Категория: IoT и системы безопасности\n"
            "Количество: 168 наблюдений (почасовые за неделю)\n"
            "Диапазон: от 1 до 80 срабатываний\n"
            "Особенность: Суточный цикл активности\n"
            "Рекомендуемое число нейронов: 12-18\n"
            "Применение: Обнаружение аномалий в системах безопасности\n"
            "Уровень сложности: Низкая\n"
        )
        desc_label = tk.Label(desc_frame, text=desc_text, justify=tk.LEFT, anchor='w', wraplength=500)
        desc_label.pack(fill=tk.X, padx=5, pady=5)

        # Кнопки для атрибутов и классов
        attr_button = ttk.Button(info_frame, text="Показать список атрибутов", command=self.show_attributes)
        attr_button.pack(pady=2)
        class_button = ttk.Button(info_frame, text="Показать список названий классов", command=self.show_classes)
        class_button.pack(pady=2)

        # Кнопка для копирования листинга
        copy_button = ttk.Button(info_frame, text="Копировать листинг программы", command=self.copy_code)
        copy_button.pack(pady=10)

    def show_attributes(self):
        """Показывает список атрибутов."""
        attr_text = (
            "Список атрибутов:\n"
            "- Глубина погружения (n): 6\n"
            "- Горизонт прогнозирования (m): 1\n"
            "- Периодичность: почасовая\n"
            "- Количество признаков (входных значений): n=6\n"
            "- Целевая переменная: следующее значение после окна\n"
        )
        messagebox.showinfo("Список атрибутов", attr_text)

    def show_classes(self):
        """Показывает список названий классов."""
        class_text = (
            "Список названий классов:\n"
            "- Класс 1: Низкая активность (значения < 25)\n"
            "- Класс 2: Нормальная активность (25 <= значения <= 50)\n"
            "- Класс 3: Высокая активность (значения > 50)\n"
        )
        messagebox.showinfo("Список названий классов", class_text)

    def setup_forecast_frame(self, parent):
        """Создает фрейм для прогноза."""
        forecast_frame = ttk.LabelFrame(parent, text="Прогноз на следующий шаг")
        forecast_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(forecast_frame, text="Введите последние n значений (через запятую):").pack(anchor='w', padx=5, pady=5)
        self.forecast_input_var = tk.StringVar()
        ttk.Entry(forecast_frame, textvariable=self.forecast_input_var, width=50).pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(forecast_frame, text="Спрогнозировать", command=self.predict_next_step).pack(pady=5)

        self.forecast_result_label = tk.Label(forecast_frame, text="Результат появится здесь.")
        self.forecast_result_label.pack(pady=5)

    def log_message(self, message):
        """Добавляет сообщение в журнал."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def browse_dataset(self):
        """Открывает диалог выбора файла датасета."""
        filename = filedialog.askopenfilename(
            title="Выберите CSV файл датасета",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.dataset_path_var.set(filename)
            # Обновляем Combobox, если файл в списке
            if filename in self.dataset_list:
                self.dataset_combobox_var.set(filename)

    def on_dataset_combo_change(self, event):
        """Обновляет dataset_path_var при выборе из Combobox и подставляет рекомендуемые параметры."""
        selected_file = self.dataset_combobox_var.get()
        self.dataset_path_var.set(selected_file)

        if self.datasets_params_df is not None:
            try:
                # Извлекаем ID датасета из названия файла
                dataset_id = int(selected_file.split('_')[1])
                row = self.datasets_params_df[self.datasets_params_df['Dataset_ID'] == dataset_id]
                if not row.empty:
                    recommended_depth = row['Depth_n'].iloc[0]
                    recommended_neurons = row['Recommended_Hidden_Neurons'].iloc[0]
                    # Предположим, что в колонке Recommended_Hidden_Neurons диапазон, например, "12-18"
                    # Извлекаем первое число как минимальное рекомендуемое
                    try:
                        min_neurons = int(recommended_neurons.split('-')[0])
                    except (ValueError, AttributeError):
                        min_neurons = recommended_neurons
                    self.depth_var.set(recommended_depth)
                    self.hidden_neurons_var.set(min_neurons)
                    self.log_message(f"Подставлены рекомендуемые параметры: Depth={recommended_depth}, Hidden Neurons={min_neurons} для {selected_file}")
            except (ValueError, IndexError):
                # Если не удалось извлечь ID, игнорируем
                pass

    def load_and_process_data(self):
        """Загружает и обрабатывает данные."""
        try:
            path = self.dataset_path_var.get()
            depth = self.depth_var.get()
            activation = self.activation_var.get()
            use_tsu = self.use_tsu_var.get()
            # Выбираем диапазон нормализации в зависимости от активации
            scaler_range = (-1, 1) if activation == 'tanh' else (0, 1)

            if not os.path.exists(path):
                messagebox.showerror("Ошибка", f"Файл не найден: {path}")
                return

            self.log_message(f"Загрузка и обработка данных из {path}...")
            self.data_processor = TimeSeriesDatasetProcessor(path, depth, scaler_range=scaler_range, use_tsu=use_tsu)
            self.data_processor.load_and_preprocess()

            # Проверка данных на NaN и бесконечность
            if np.any(np.isnan(self.data_processor.X)) or np.any(np.isinf(self.data_processor.X)):
                messagebox.showwarning("Предупреждение", "Данные содержат NaN или бесконечность. Попробуйте изменить параметры или очистить данные.")
                self.log_message("Предупреждение: Данные содержат NaN или бесконечность.")
            elif np.any(np.isnan(self.data_processor.y)) or np.any(np.isinf(self.data_processor.y)):
                messagebox.showwarning("Предупреждение", "Целевые значения содержат NaN или бесконечность. Попробуйте изменить параметры или очистить данные.")
                self.log_message("Предупреждение: Целевые значения содержат NaN или бесконечность.")

            self.log_message("Данные успешно загружены и обработаны.")
            self.update_results_display(f"Данные загружены. Размер исходного ряда: {len(self.data_processor.data)}, "
                                        f"Размер X_train: {self.data_processor.X_train.shape}, "
                                        f"Размер X_test: {self.data_processor.X_test.shape}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            self.log_message(f"Ошибка: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при загрузке данных: {e}")

    def train_model(self):
        """Создает и обучает модель."""
        if self.data_processor is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите и обработайте данные.")
            return

        try:
            self.log_message("Создание модели...")
            input_size = self.data_processor.X_train.shape[1] # Глубина погружения
            output_size = 1 if self.task_var.get() == 'regression' else 3 # 3 класса для классификации
            hidden_size = self.hidden_neurons_var.get()
            activation = self.activation_var.get()
            lr = self.learning_rate_var.get()
            optimizer = self.optimizer_var.get()
            use_dropout = self.use_dropout_var.get()
            dropout_rate = self.dropout_rate_var.get()
            use_batch_norm = self.use_batch_norm_var.get()
            task = self.task_var.get()

            self.forecaster = MLPForecaster(input_size, hidden_size, output_size, activation, learning_rate=lr, optimizer=optimizer, use_dropout=use_dropout, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm, task=task)
            self.forecaster.build_model()

            self.log_message("Обучение модели...")
            self.current_history = self.forecaster.train(
                self.data_processor.X_train, self.data_processor.y_train if task == 'regression' else self.data_processor.y_train_class,
                self.data_processor.X_test, self.data_processor.y_test if task == 'regression' else self.data_processor.y_test_class, # Используем классы для валидации
                progress_callback=self.update_progress
            )
            self.log_message("Обучение завершено.")
            self.progress_var.set(0) # Сбросить прогресс

        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            self.log_message(f"Ошибка: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при обучении модели: {e}")

    def update_progress(self, value):
        """Обновляет значение прогресс-бара."""
        self.progress_var.set(value)
        self.root.update_idletasks() # Обновить интерфейс

    def evaluate_model(self):
        """Оценивает модель и выводит метрики."""
        if self.forecaster is None:
            messagebox.showwarning("Предупреждение", "Сначала обучите модель.")
            return

        try:
            self.log_message("Оценка модели на тестовой выборке...")
            task = self.task_var.get()
            if task == 'regression':
                eval_results = self.forecaster.evaluate(self.data_processor.X_test, self.data_processor.y_test)
                # Восстановление значений в исходный масштаб
                y_test_actual = self.data_processor.inverse_transform(self.data_processor.y_test)
                y_pred_actual = self.data_processor.inverse_transform(eval_results['y_pred'])

                # Расчет метрик на исходном масштабе
                mse_actual = mean_squared_error(y_test_actual, y_pred_actual)
                mae_actual = mean_absolute_error(y_test_actual, y_pred_actual)
                rmse_actual = np.sqrt(mse_actual)

                # Добавляем метрики на исходном масштабе в результаты
                eval_results['MSE_actual'] = mse_actual
                eval_results['MAE_actual'] = mae_actual
                eval_results['RMSE_actual'] = rmse_actual
                eval_results['y_test_actual'] = y_test_actual
                eval_results['y_pred_actual'] = y_pred_actual

                result_str = (
                    f"--- Результаты оценки модели (Регрессия) ---\n"
                    f"Параметры: Нейронов={self.hidden_neurons_var.get()}, Активация={self.activation_var.get()}, Оптимизатор={self.optimizer_var.get()}\n"
                    f"Тестовая MSE (норм.): {eval_results['MSE']:.6f}\n"
                    f"Тестовая MAE (норм.): {eval_results['MAE']:.6f}\n"
                    f"Тестовая RMSE (норм.): {eval_results['RMSE']:.6f}\n"
                    f"Тестовая MSE (исх.): {eval_results['MSE_actual']:.6f}\n"
                    f"Тестовая MAE (исх.): {eval_results['MAE_actual']:.6f}\n"
                    f"Тестовая RMSE (исх.): {eval_results['RMSE_actual']:.6f}\n"
                )
            elif task == 'classification':
                eval_results = self.forecaster.evaluate(self.data_processor.X_test, None, y_test_class=self.data_processor.y_test_class)
                result_str = (
                    f"--- Результаты оценки модели (Классификация) ---\n"
                    f"Параметры: Нейронов={self.hidden_neurons_var.get()}, Активация={self.activation_var.get()}, Оптимизатор={self.optimizer_var.get()}\n"
                    f"Тестовая Accuracy: {eval_results['Accuracy']:.6f}\n"
                    f"Тестовая Precision (macro): {eval_results['Precision']:.6f}\n"
                    f"Тестовая Recall (macro): {eval_results['Recall']:.6f}\n"
                    f"Тестовая F1-score (macro): {eval_results['F1']:.6f}\n"
                    f"Матрица ошибок:\n{eval_results['Confusion_Matrix']}\n"
                )

            # Добавляем описание датасета 15
            description = (
                f"\n--- Описание датасета (Вариант 15) ---\n"
                f"Название: Срабатывания датчиков безопасности\n"
                f"Категория: IoT и системы безопасности\n"
                f"Количество наблюдений: 168 (почасовые за неделю)\n"
                f"Диапазон значений: от 1 до 80 срабатываний\n"
                f"Особенность: Суточный цикл активности\n"
                f"Рекомендуемое число нейронов: 12-18\n"
                f"Применение: Обнаружение аномалий в системах безопасности\n"
                f"Уровень сложности: Низкий\n"
            )
            result_str += description
            self.update_results_display(result_str)

            self.current_results = eval_results  # Сохраняем все результаты

        except Exception as e:
            logger.error(f"Ошибка при оценке модели: {e}")
            self.log_message(f"Ошибка: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при оценке модели: {e}")

    def update_results_display(self, text):
        """Обновляет текстовое поле с результатами."""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state='disabled')

    def show_plots_in_new_window(self):
        """Открывает окно с графиками в отдельном окне."""
        if not self.current_history or not self.current_results:
            messagebox.showwarning("Предупреждение", "Сначала обучите и оцените модель.")
            return

        # Создаем новое окно
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Графики модели")
        plot_window.geometry("1200x800")

        # Создаем Canvas с Scrollbar для нового окна
        canvas_container = tk.Frame(plot_window)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        plot_canvas = tk.Canvas(canvas_container)
        plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=plot_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        plot_canvas.configure(yscrollcommand=scrollbar.set)

        plot_inner_frame = tk.Frame(plot_canvas)
        plot_canvas.create_window((0, 0), window=plot_inner_frame, anchor="nw")

        def on_configure(event):
            plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))

        plot_inner_frame.bind("<Configure>", on_configure)

        # Создаем matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Результаты модели (Нейронов: {self.hidden_neurons_var.get()}, Активация: {self.activation_var.get()}, Оптимизатор: {self.optimizer_var.get()})')

        # 1. История обучения (Loss)
        axes[0, 0].plot(self.current_history.history['loss'], label='Обучающая Loss')
        if 'val_loss' in self.current_history.history:
            axes[0, 0].plot(self.current_history.history['val_loss'], label='Валидационная Loss')
        axes[0, 0].set_title('История обучения (Loss)')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. История MAE или Accuracy
        task = self.task_var.get()
        if task == 'regression':
            metric_name = 'mae'
            metric_label = 'MAE'
        else: # classification
            metric_name = 'accuracy'
            metric_label = 'Accuracy'
        axes[0, 1].plot(self.current_history.history[metric_name], label=f'Обучающий {metric_label}')
        if f'val_{metric_name}' in self.current_history.history:
            axes[0, 1].plot(self.current_history.history[f'val_{metric_name}'], label=f'Валидационный {metric_label}')
        axes[0, 1].set_title(f'История {metric_label}')
        axes[0, 1].set_xlabel('Эпоха')
        axes[0, 1].set_ylabel(metric_label)
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. Предсказания vs Реальные (в исходном масштабе, если регрессия)
        if task == 'regression' and 'y_test_actual' in self.current_results:
            y_test_actual = self.current_results['y_test_actual']
            y_pred_actual = self.current_results['y_pred_actual']
            axes[1, 0].plot(y_test_actual, label='Реальные значения', alpha=0.7)
            axes[1, 0].plot(y_pred_actual, label='Предсказания', alpha=0.7)
            axes[1, 0].set_title('Предсказания vs Реальные (исх. масштаб)')
            axes[1, 0].set_xlabel('Индекс')
            axes[1, 0].set_ylabel('Значение')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'График недоступен для задачи классификации', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Предсказания vs Реальные (исх. масштаб)')

        # 4. Остатки (в исходном масштабе, если регрессия)
        if task == 'regression' and 'y_test_actual' in self.current_results:
            residuals = self.current_results['y_test_actual'].flatten() - self.current_results['y_pred_actual'].flatten()
            axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_title('Остатки (Реальные - Предсказанные)')
            axes[1, 1].set_xlabel('Индекс')
            axes[1, 1].set_ylabel('Остаток')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'График недоступен для задачи классификации', horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Остатки (Реальные - Предсказанные)')

        plt.tight_layout()

        # Встраиваем figure в tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_inner_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Обновляем scrollregion
        plot_canvas.update_idletasks()
        plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))

        # Добавляем кнопку "Сохранить график" в новое окно
        save_plot_button = ttk.Button(plot_window, text="Сохранить график как PNG", command=lambda: self.save_current_plot(fig))
        save_plot_button.pack(pady=5)

    def save_current_plot(self, fig):
        """Сохраняет текущий matplotlib figure как PNG."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            try:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.log_message(f"График сохранен как {filename}")
                messagebox.showinfo("Сохранение", f"График успешно сохранен как {filename}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении графика: {e}")
                messagebox.showerror("Ошибка", f"Ошибка при сохранении графика: {e}")

    def save_results(self):
        """Сохраняет результаты в файл."""
        if not self.current_results:
            messagebox.showwarning("Предупреждение", "Нет результатов для сохранения.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                # Подготовим данные для сохранения
                save_data = {
                    "parameters": {
                        "dataset_path": self.dataset_path_var.get(),
                        "task": self.task_var.get(),
                        "depth": self.depth_var.get(),
                        "hidden_neurons": self.hidden_neurons_var.get(),
                        "activation": self.activation_var.get(),
                        "optimizer": self.optimizer_var.get(),
                        "learning_rate": self.learning_rate_var.get(),
                        "max_error": self.max_error_var.get(),
                        "use_dropout": self.use_dropout_var.get(),
                        "dropout_rate": self.dropout_rate_var.get(),
                        "use_batch_norm": self.use_batch_norm_var.get(),
                    },
                    "metrics": self.current_results
                }
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=4, ensure_ascii=False)
                self.log_message(f"Результаты сохранены в {filename}")
                messagebox.showinfo("Сохранение", f"Результаты успешно сохранены в {filename}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении результатов: {e}")
                messagebox.showerror("Ошибка", f"Ошибка при сохранении результатов: {e}")

    def add_to_table(self):
        """Добавляет текущие результаты в таблицу."""
        if not self.current_results:
            messagebox.showwarning("Предупреждение", "Нет результатов для добавления.")
            return

        try:
            # Получаем параметры из GUI
            dataset_name = os.path.basename(self.dataset_path_var.get()).replace('.csv', '')
            task = self.task_var.get()
            activation = self.activation_var.get()
            optimizer = self.optimizer_var.get()
            hidden_neurons = self.hidden_neurons_var.get()
            depth = self.depth_var.get()
            uses_dropout = self.use_dropout_var.get()
            dropout_rate = self.dropout_rate_var.get()
            uses_batch_norm = self.use_batch_norm_var.get()

            # Получаем метрики
            row_data = [dataset_name, task, activation, optimizer, hidden_neurons, depth, uses_dropout, dropout_rate, uses_batch_norm]

            if task == 'regression':
                row_data.extend([
                    f"{self.current_results.get('MSE', 'N/A'):.6f}",
                    f"{self.current_results.get('MAE', 'N/A'):.6f}",
                    f"{self.current_results.get('RMSE', 'N/A'):.6f}",
                    f"{self.current_results.get('MSE_actual', 'N/A'):.6f}",
                    f"{self.current_results.get('MAE_actual', 'N/A'):.6f}",
                    f"{self.current_results.get('RMSE_actual', 'N/A'):.6f}",
                    "N/A", "N/A" # Для Accuracy и F1 в регрессии
                ])
            elif task == 'classification':
                row_data.extend([
                    "N/A", "N/A", "N/A", # Для MSE, MAE, RMSE в классификации
                    "N/A", "N/A", "N/A", # Для MSE_actual, MAE_actual, RMSE_actual в классификации
                    f"{self.current_results.get('Accuracy', 'N/A'):.6f}",
                    f"{self.current_results.get('F1', 'N/A'):.6f}"
                ])

            # Добавляем в таблицу
            self.results_tree.insert('', 'end', values=row_data)

            self.log_message(f"Результаты для {dataset_name} (task={task}) добавлены в таблицу.")

        except Exception as e:
            logger.error(f"Ошибка при добавлении в таблицу: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при добавлении в таблицу: {e}")

    def export_table(self):
        """Экспортирует таблицу результатов в CSV файл."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Заголовки
                    headers = [self.results_tree.heading(col, option="text") for col in self.results_tree["columns"]]
                    writer.writerow(headers)
                    # Данные
                    for item in self.results_tree.get_children():
                        row = self.results_tree.item(item)['values']
                        writer.writerow(row)
                self.log_message(f"Таблица результатов сохранена в {filename}")
                messagebox.showinfo("Экспорт", f"Таблица успешно сохранена в {filename}")
            except Exception as e:
                logger.error(f"Ошибка при экспорте таблицы: {e}")
                messagebox.showerror("Ошибка", f"Ошибка при экспорте таблицы: {e}")

    def show_training_samples(self):
        """Показывает первые 10 примеров обучающей выборки."""
        if self.data_processor is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите и обработайте данные.")
            return

        try:
            # Берем первые 10 строк X_train и y_train, но не больше, чем есть
            num_samples = min(10, len(self.data_processor.X_train))
            sample_X_norm = self.data_processor.X_train[:num_samples]
            sample_y_norm = self.data_processor.y_train[:num_samples]

            # Восстанавливаем в исходный масштаб
            sample_X_actual = self.data_processor.inverse_transform(sample_X_norm)
            sample_y_actual = self.data_processor.inverse_transform(sample_y_norm.reshape(-1, 1)).flatten()

            # Формируем строку для вывода
            sample_str = f"Первые {num_samples} примеров обучающей выборки (нормализованные):\n"
            sample_str += "X (история) -> y (прогноз)\n"
            for i in range(len(sample_X_norm)):
                x_str = ", ".join([f"{x:.6f}" for x in sample_X_norm[i]])
                y_str = f"{sample_y_norm[i]:.6f}"
                sample_str += f"[{x_str}] -> [{y_str}]\n"

            sample_str += f"\nПервые {num_samples} примеров обучающей выборки (исходный масштаб):\n"
            sample_str += "X (история) -> y (прогноз)\n"
            for i in range(len(sample_X_actual)):
                x_str = ", ".join([f"{x:.2f}" for x in sample_X_actual[i]])
                y_str = f"{sample_y_actual[i]:.2f}"
                sample_str += f"[{x_str}] -> [{y_str}]\n"

            self.update_results_display(sample_str)
            self.log_message(f"Первые {num_samples} примеров обучающей выборки показаны (норм. и исх. масштаб).")

        except Exception as e:
            logger.error(f"Ошибка при показе примеров: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при показе примеров: {e}")

    def copy_code(self):
        """Копирует весь код программы в буфер обмена."""
        try:
            # Прочитаем содержимое текущего файла
            with open(__file__, 'r', encoding='utf-8') as f:
                code = f.read()
            self.root.clipboard_clear()
            self.root.clipboard_append(code)
            self.log_message("Листинг программы скопирован в буфер обмена.")
            messagebox.showinfo("Копирование", "Листинг программы скопирован в буфер обмена.")
        except Exception as e:
            logger.error(f"Ошибка при копировании кода: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при копировании кода: {e}")

    def show_about(self):
        """Показывает информацию о программе."""
        about_text = (
            "Программа для прогнозирования временных рядов\n"
            "Лабораторная работа №2 по дисциплине \"Основы ИИ\"\n"
            "Вариант 15: Срабатывания датчиков безопасности\n"
            "Автор: Колосов Станислав\n"
            "Дата: 2025"
        )
        messagebox.showinfo("О программе", about_text)

    def save_config(self):
        """Сохраняет текущие параметры в файл JSON."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                config = {
                    "dataset_path": self.dataset_path_var.get(),
                    "dataset_combobox": self.dataset_combobox_var.get(),
                    "depth": self.depth_var.get(),
                    "hidden_neurons": self.hidden_neurons_var.get(),
                    "activation": self.activation_var.get(),
                    "optimizer": self.optimizer_var.get(),
                    "learning_rate": self.learning_rate_var.get(),
                    "max_error": self.max_error_var.get(),
                    "use_dropout": self.use_dropout_var.get(),
                    "dropout_rate": self.dropout_rate_var.get(),
                    "use_batch_norm": self.use_batch_norm_var.get(),
                    "task": self.task_var.get(),
                    "use_tsu": self.use_tsu_var.get(),
                    "hidden_neurons_list": self.hidden_neurons_list_var.get(),
                    "activation_list": self.activation_list_var.get(),
                    "depth_list": self.depth_list_var.get(),
                    "dataset_list": self.dataset_list_var.get(),
                }
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                self.log_message(f"Конфигурация сохранена в {filename}")
                messagebox.showinfo("Сохранение", f"Конфигурация успешно сохранена в {filename}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении конфигурации: {e}")
                messagebox.showerror("Ошибка", f"Ошибка при сохранении конфигурации: {e}")

    def load_config(self):
        """Загружает параметры из файла JSON."""
        filename = filedialog.askopenfilename(
            title="Выберите файл конфигурации",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # Устанавливаем значения переменных
                self.dataset_path_var.set(config.get("dataset_path", "dataset_15_Sensor_Events.csv"))
                self.dataset_combobox_var.set(config.get("dataset_combobox", "dataset_15_Sensor_Events.csv"))
                self.depth_var.set(config.get("depth", 6))
                self.hidden_neurons_var.set(config.get("hidden_neurons", 12))
                self.activation_var.set(config.get("activation", "tanh"))
                self.optimizer_var.set(config.get("optimizer", "SGD"))
                self.learning_rate_var.set(config.get("learning_rate", 0.05))
                self.max_error_var.set(config.get("max_error", 0.005))
                self.use_dropout_var.set(config.get("use_dropout", False))
                self.dropout_rate_var.set(config.get("dropout_rate", 0.1))
                self.use_batch_norm_var.set(config.get("use_batch_norm", False))
                self.task_var.set(config.get("task", "regression"))
                self.use_tsu_var.set(config.get("use_tsu", HAS_TSU))
                self.hidden_neurons_list_var.set(config.get("hidden_neurons_list", "8, 12, 18"))
                self.activation_list_var.set(config.get("activation_list", "tanh, sigmoid"))
                self.depth_list_var.set(config.get("depth_list", "4, 6"))
                self.dataset_list_var.set(config.get("dataset_list", "dataset_15_Sensor_Events.csv"))

                self.log_message(f"Конфигурация загружена из {filename}")
                messagebox.showinfo("Загрузка", f"Конфигурация успешно загружена из {filename}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке конфигурации: {e}")
                messagebox.showerror("Ошибка", f"Ошибка при загрузке конфигурации: {e}")

    def run_experiment_series(self):
        """Запускает серию экспериментов."""
        # if self.data_processor is None: # Убираем проверку, так как будем перезагружать данные
        #    messagebox.showwarning("Предупреждение", "Сначала загрузите и обработайте данные.")
        #    return

        try:
            # Получаем списки параметров из GUI
            hidden_neurons_str = self.hidden_neurons_list_var.get()
            activation_str = self.activation_list_var.get()
            depth_str = self.depth_list_var.get()
            dataset_str = self.dataset_list_var.get() # Новый список

            # Преобразуем строки в списки
            hidden_neurons_list = [int(x.strip()) for x in hidden_neurons_str.split(',') if x.strip().isdigit()]
            activation_list = [x.strip() for x in activation_str.split(',') if x.strip() in ["tanh", "sigmoid"]]
            depth_list = [int(x.strip()) for x in depth_str.split(',') if x.strip().isdigit()]
            dataset_list = [x.strip() for x in dataset_str.split(',') if x.strip() in self.dataset_list] # Проверяем на валидность

            if not hidden_neurons_list or not activation_list or not depth_list or not dataset_list: # Добавляем проверку dataset_list
                messagebox.showerror("Ошибка", "Списки параметров для экспериментов некорректны или пусты.")
                return

            self.log_message(f"Запуск серии экспериментов.")
            self.log_message(f"Нейроны: {hidden_neurons_list}, Активации: {activation_list}, Глубины: {depth_list}, Датасеты: {dataset_list}")

            # Запускаем в отдельном потоке
            thread = threading.Thread(target=self._run_experiments_thread, args=(hidden_neurons_list, activation_list, depth_list, dataset_list))
            thread.start()

        except Exception as e:
            logger.error(f"Ошибка при запуске серии экспериментов: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при запуске серии экспериментов: {e}")

    def _run_experiments_thread(self, hidden_neurons_list, activation_list, depth_list, dataset_list):
        """Внутренняя функция для выполнения экспериментов в потоке."""
        combinations = list(itertools.product(dataset_list, hidden_neurons_list, activation_list, depth_list))
        total = len(combinations)
        self.log_message(f"Всего комбинаций: {total}")

        for i, (ds_file, hn, act, d) in enumerate(combinations):
            self.log_message(f"Эксперимент {i+1}/{total}: Датасет={ds_file}, Нейроны={hn}, Активация={act}, Глубина={d}")

            # Обновляем параметры
            self.dataset_path_var.set(ds_file)
            self.dataset_combobox_var.set(ds_file)
            self.hidden_neurons_var.set(hn)
            self.activation_var.set(act)
            self.depth_var.set(d)
            # Загружаем данные
            self.load_and_process_data() # Перезагрузка с новыми параметрами
            # Обучаем модель
            self.train_model()
            # Оцениваем модель
            self.evaluate_model()
            # Добавляем результаты в таблицу
            self.add_to_table()

        self.log_message("Серия экспериментов завершена.")

    def save_trained_model(self):
        """Сохраняет обученную модель Keras."""
        if self.forecaster is None or self.forecaster.model is None:
            messagebox.showwarning("Предупреждение", "Сначала обучите модель.")
            return

        directory = filedialog.askdirectory(title="Выберите папку для сохранения модели")
        if directory:
            try:
                model_path = os.path.join(directory, "my_trained_model")
                self.forecaster.model.save(model_path)
                self.log_message(f"Модель сохранена в {model_path}")
                messagebox.showinfo("Сохранение", f"Модель успешно сохранена в {model_path}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении модели: {e}")
                messagebox.showerror("Ошибка", f"Ошибка при сохранении модели: {e}")

    def save_training_history(self):
        """Сохраняет историю обучения в JSON файл."""
        if self.current_history is None:
            messagebox.showwarning("Предупреждение", "Нет истории обучения для сохранения.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.current_history.history, f, indent=4, ensure_ascii=False)
                self.log_message(f"История обучения сохранена в {filename}")
                messagebox.showinfo("Сохранение", f"История обучения успешно сохранена в {filename}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении истории: {e}")
                messagebox.showerror("Ошибка", f"Ошибка при сохранении истории: {e}")

    def load_training_history(self):
        """Загружает историю обучения из JSON файла."""
        filename = filedialog.askopenfilename(
            title="Выберите файл истории обучения",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    history_dict = json.load(f)
                # Создаем фейковый объект history
                class DummyHistory:
                    def __init__(self, hist_dict):
                        self.history = hist_dict
                self.current_history = DummyHistory(history_dict)
                self.log_message(f"История обучения загружена из {filename}")
                messagebox.showinfo("Загрузка", f"История обучения успешно загружена из {filename}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке истории: {e}")
                messagebox.showerror("Ошибка", f"Ошибка при загрузке истории: {e}")

    def predict_next_step(self):
        """Прогнозирует следующий шаг на основе введенных данных."""
        if self.forecaster is None or self.forecaster.model is None:
            messagebox.showwarning("Предупреждение", "Сначала обучите модель.")
            return

        try:
            input_str = self.forecast_input_var.get()
            if not input_str:
                messagebox.showwarning("Предупреждение", "Введите значения для прогноза.")
                return

            input_values = [float(x.strip()) for x in input_str.split(',')]
            if len(input_values) != self.depth_var.get():
                messagebox.showerror("Ошибка", f"Количество введенных значений ({len(input_values)}) не соответствует глубине погружения ({self.depth_var.get()}).")
                return

            # Нормализуем введенные значения с использованием текущего scaler'а
            # Это может быть неточно, если введенные значения вне диапазона обучающих данных
            # Лучше использовать scaler, обученный на общей выборке, или вручную указать min/max
            # Пока используем текущий scaler
            scaler = self.data_processor.scaler
            normalized_input = scaler.transform(np.array(input_values).reshape(-1, 1)).flatten()
            X_new = normalized_input.reshape(1, -1) # Формируем батч (1, depth)

            # Предсказание
            y_pred_norm = self.forecaster.predict(X_new)
            # Денормализуем результат
            y_pred_actual = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()[0]

            # Класс
            y_pred_class = self.data_processor.value_to_class(y_pred_actual)

            result_text = f"Прогноз (норм.): {y_pred_norm[0][0]:.6f}\nПрогноз (исх.): {y_pred_actual:.2f}\nКласс: {y_pred_class[0]}"
            self.forecast_result_label.config(text=result_text)
            self.log_message(f"Прогноз на следующий шаг: {y_pred_actual:.2f} (Класс: {y_pred_class[0]})")

        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный формат ввода. Введите числа, разделенные запятыми.")
        except Exception as e:
            logger.error(f"Ошибка при прогнозировании: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при прогнозировании: {e}")


def main():
    root = tk.Tk()
    app = FuzzyLogicApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
