"""
time_series_utils.py

Вспомогательный модуль для обработки временных рядов.
Содержит класс TimeSeriesDataset для загрузки, нормализации,
создания окон и разделения данных.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesDataset:
    """
    Класс для обработки временных рядов: нормализация, создание окон, разделение выборки.
    """

    def __init__(self, filepath=None, data=None, depth=5, horizon=1, feature_range=(-1, 1)):
        """
        Инициализация процессора данных.

        Args:
            filepath (str, optional): Путь к CSV-файлу датасета.
                                      Должен содержать колонку 'Value' или 'Output'.
            data (array-like, optional): Непосредственно временной ряд (если filepath не указан).
            depth (int): Глубина погружения (n) для метода окон.
            horizon (int): Горизонт прогнозирования (m).
            feature_range (tuple): Диапазон для нормализации (min, max).
        """
        if filepath:
            df = pd.read_csv(filepath)
            # Предполагаем, что колонка с исходными данными называется 'Value' или 'Output'
            # Можно адаптировать под конкретный формат файла
            if 'Value' in df.columns:
                self.raw_data = df['Value'].values.astype(np.float32)
            elif 'Output' in df.columns:
                # Для файлов с готовыми окнами, извлекаем 'Output' как исходный ряд
                # Это менее надежно, чем использовать 'Value' из all_datasets_combined
                self.raw_data = df['Output'].values.astype(np.float32)
                # Восстанавливаем начало ряда из первого окна Input
                first_window_inputs = df.iloc[0, :depth].values.astype(np.float32)
                self.raw_data = np.concatenate([first_window_inputs, self.raw_data])
            else:
                raise ValueError(f"Файл {filepath} не содержит колонки 'Value' или 'Output'.")
        elif data is not None:
            self.raw_data = np.array(data, dtype=np.float32)
        else:
            raise ValueError("Необходимо указать либо filepath, либо data.")

        self.depth = depth
        self.horizon = horizon
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.normalized_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_process(self):
        """
        Выполняет полную предварительную обработку:
        нормализация, создание окон, разделение на обучающую и тестовую выборки.
        """
        # Нормализация
        self.normalized_data = self.scaler.fit_transform(self.raw_data.reshape(-1, 1)).flatten()

        # Создание окон
        self.X, self.y = self.create_windows(self.normalized_data, self.depth, self.horizon)

        # Разделение на обучающую и тестовую выборки (70/30)
        split_idx = int(len(self.X) * 0.7)
        self.X_train = self.X[:split_idx]
        self.X_test = self.X[split_idx:]
        self.y_train = self.y[:split_idx]
        self.y_test = self.y[split_idx:]

    def create_windows(self, data, depth, horizon):
        """
        Создает окна для обучения из одномерного временного ряда.

        Args:
            data (np.array): Нормализованный одномерный массив.
            depth (int): Глубина погружения (n).
            horizon (int): Горизонт прогнозирования (m).

        Returns:
            tuple: (X, y) - массивы признаков и целевых значений.
        """
        X, y = [], []
        for i in range(len(data) - depth - horizon + 1):
            X.append(data[i:i + depth])
            y.append(data[i + depth:i + depth + horizon][0])  # Только первое значение после окна
        return np.array(X), np.array(y)

    def split_data(self, train_ratio=0.7):
        """
        Разделение данных на обучающую и тестовую выборки.
        (Альтернатива вызову load_and_process, если данные уже созданы)
        Args:
            train_ratio (float): Доля обучающей выборки.
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.X is None or self.y is None:
            raise ValueError("Данные (X, y) не созданы. Вызовите create_windows или load_and_process.")
        split_idx = int(len(self.X) * train_ratio)
        self.X_train = self.X[:split_idx]
        self.X_test = self.X[split_idx:]
        self.y_train = self.y[:split_idx]
        self.y_test = self.y[split_idx:]
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_features_and_targets(self):
        """
        Возвращает созданные окна (X, y).
        """
        if self.X is None or self.y is None:
            raise ValueError("Данные (X, y) не созданы. Вызовите create_windows или load_and_process.")
        return self.X, self.y

    def inverse_normalize(self, value):
        """
        Восстанавливает значение из нормализованного масштаба в исходный.
        """
        return self.scaler.inverse_transform(np.array(value).reshape(-1, 1))

    def get_raw_data(self):
        """
        Возвращает исходные данные.
        """
        return self.raw_data

    def get_normalized_data(self):
        """
        Возвращает нормализованные данные.
        """
        return self.normalized_data
