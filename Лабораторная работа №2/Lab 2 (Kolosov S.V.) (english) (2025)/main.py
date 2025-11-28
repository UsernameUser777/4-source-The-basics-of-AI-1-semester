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

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import from time_series_utils.py
try:
    import time_series_utils
    HAS_TSU = True
    logger.info("time_series_utils.py imported.")
except ImportError:
    HAS_TSU = False
    logger.info("time_series_utils.py not found, using built-in implementation.")


class TimeSeriesDatasetProcessor:
    """
    Class for processing time series data: loading, normalization, creating windows, splitting data.
    Can use the external time_series_utils.py module or its own implementation.
    """
    def __init__(self, dataset_path, depth, scaler_range=(-1, 1), use_tsu=HAS_TSU):
        """
        Initialize the data processor.

        Args:
            dataset_path (str): Path to the CSV dataset file.
            depth (int): Depth of embedding (n) for the window method.
            scaler_range (tuple): Range for normalization (min, max).
            use_tsu (bool): Whether to use time_series_utils.py.
        """
        logger.info(f"Initializing data processor for {dataset_path} (use_tsu={use_tsu})")
        self.dataset_path = dataset_path
        self.depth = depth
        self.scaler_range = scaler_range
        self.use_tsu = use_tsu
        self.scaler = MinMaxScaler(feature_range=scaler_range)
        self.data = None  # Original time series
        self.normalized_data = None  # Normalized series
        self.X = None  # Input features (windows)
        self.y = None  # Target values (next value after window)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_class = None # Target values for classification
        self.y_test_class = None
        self.scaler_used = scaler_range # For info

        if self.use_tsu:
            # Use time_series_utils
            self.ts_dataset = time_series_utils.TimeSeriesDataset(
                filepath=dataset_path,
                depth=depth,
                horizon=1,
                feature_range=scaler_range
            )
            self.ts_dataset.load_and_process()
            self.X, self.y = self.ts_dataset.get_features_and_targets()
            self.scaler = self.ts_dataset.scaler # Use scaler from TSU

    def load_and_preprocess(self):
        """
        Load data and perform full preprocessing:
        loading, normalization, creating windows, splitting into training and test sets.
        For classification tasks, also creates class labels.
        """
        logger.info(f"Loading data from {self.dataset_path}")
        if self.use_tsu:
            logger.info("Using time_series_utils for preprocessing.")
            # Already done in __init__ if use_tsu=True
            self.X, self.y = self.ts_dataset.get_features_and_targets()
            self.X_train, self.X_test, self.y_train, self.y_test = self.ts_dataset.split_data(train_ratio=0.7)
            self.data = self.ts_dataset.raw_data.flatten() # Extract raw data from TSU
        else:
            logger.info("Using built-in implementation for preprocessing.")
            # Try to load the original time series from all_datasets_combined.csv
            all_data_path = self.dataset_path.replace("dataset_15_Sensor_Events.csv", "all_datasets_combined.csv")
            if os.path.exists(all_data_path):
                logger.info(f"Loading original series from {all_data_path}")
                full_df = pd.read_csv(all_data_path)
                # Extract dataset ID from filename
                try:
                    dataset_id = int(self.dataset_path.split('_')[1])
                except (IndexError, ValueError):
                    dataset_id = 15 # Default
                dataset_df = full_df[full_df['Dataset_ID'] == dataset_id].copy()
                self.data = dataset_df['Value'].values.astype(float)
            else:
                # If all_datasets_combined.csv is unavailable, reconstruct the series from the window file
                logger.info("File all_datasets_combined.csv not found, reconstructing series from CSV windows")
                df_windows = pd.read_csv(self.dataset_path)
                # Reconstruction: first 'depth' values from Input_1 ... Input_depth + all Output
                first_inputs = df_windows.iloc[0, :self.depth].values
                outputs = df_windows['Output'].values
                # Concatenate: [Input_1, Input_2, ..., Input_depth, Output_1, Output_2, ...]
                self.data = np.concatenate([first_inputs, outputs])
                logger.info(f"Series reconstructed. Length: {len(self.data)}")

            logger.info(f"Loaded {len(self.data)} data points.")

            # Normalize
            self.normalized_data = self.scaler.fit_transform(self.data.reshape(-1, 1)).flatten()
            logger.info("Data normalized.")

            # Create windows
            self.X, self.y = self.create_windows(self.normalized_data, self.depth, horizon=1)
            logger.info(f"Created {len(self.X)} training examples using sliding window method.")

            # Split into training and test sets (70/30)
            split_idx = int(len(self.X) * 0.7)
            self.X_train = self.X[:split_idx]
            self.X_test = self.X[split_idx:]
            self.y_train = self.y[:split_idx]
            self.y_test = self.y[split_idx:]
            logger.info(f"Split: {len(self.X_train)} training, {len(self.X_test)} test.")

        # Create classes for classification task
        # Use thresholds from datasets_description.pdf for dataset 15
        # 1: < 25, 2: 25-50, 3: > 50
        # Denormalize y_train and y_test to determine classes
        y_train_denorm = self.inverse_transform(self.y_train)
        y_test_denorm = self.inverse_transform(self.y_test)
        self.y_train_class = self.value_to_class(y_train_denorm)
        self.y_test_class = self.value_to_class(y_test_denorm)

    def value_to_class(self, values):
        """
        Convert values to classes for dataset 15.

        Args:
            values (array-like): Values to convert.

        Returns:
            np.ndarray: Array of integer class labels.
        """
        values = np.array(values).flatten()
        classes = np.zeros_like(values, dtype=int)
        classes[values < 25] = 1
        classes[(values >= 25) & (values <= 50)] = 2
        classes[values > 50] = 3
        return classes

    def create_windows(self, data, depth, horizon):
        """
        Creates windows for training from a 1D time series.
        Used if use_tsu=False.

        Args:
            data (np.array): Normalized 1D array.
            depth (int): Depth of embedding (n).
            horizon (int): Forecasting horizon (m).

        Returns:
            tuple: (X, y) - feature arrays and target values.
        """
        X, y = [], []
        for i in range(len(data) - depth - horizon + 1):
            X.append(data[i:i + depth])
            y.append(data[i + depth:i + depth + horizon][0]) # Only the first value after the window
        return np.array(X), np.array(y)

    def inverse_transform(self, value):
        """
        Revert value from normalized scale back to original.
        Uses TSU's scaler if use_tsu=True.

        Args:
            value (array-like): Normalized values.

        Returns:
            np.ndarray: Denormalized values.
        """
        if self.use_tsu:
            return self.ts_dataset.inverse_normalize(value)
        else:
            return self.scaler.inverse_transform(np.array(value).reshape(-1, 1))

class MLPForecaster:
    """
    Class for building, training, and evaluating a Multi-Layer Perceptron.
    Supports regression and classification tasks.
    """
    def __init__(self, input_size, hidden_size, output_size, activation_func, learning_rate=0.05, optimizer='SGD', use_dropout=False, dropout_rate=0.0, use_batch_norm=False, task='regression'):
        """
        Initialize the model.

        Args:
            input_size (int): Input layer dimension (embedding depth).
            hidden_size (int): Hidden layer dimension.
            output_size (int): Output layer dimension (usually 1).
            activation_func (str): Activation function ('tanh' or 'sigmoid').
            learning_rate (float): Initial learning rate.
            optimizer (str): Optimizer ('SGD', 'Adam', 'RMSprop').
            use_dropout (bool): Whether to use Dropout.
            dropout_rate (float): Dropout rate.
            use_batch_norm (bool): Whether to use BatchNormalization.
            task (str): 'regression' or 'classification'.
        """
        logger.info(f"Initializing MLPForecaster: {input_size} -> {hidden_size} ({activation_func}) -> {output_size} (task={task})")
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
        Builds the neural network architecture.
        Architecture: 2 layers (implicit input, hidden, output).
        """
        logger.info(f"Building model: {self.input_size} -> {self.hidden_size} ({self.activation_func}) -> {self.output_size} (task={self.task})")
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
            self.model.add(layers.Dense(self.output_size, activation='softmax')) # 3 classes
            loss = 'sparse_categorical_crossentropy'

        if self.optimizer_name == 'SGD':
            optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'RMSprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate) # Default

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['mae' if self.task == 'regression' else 'accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10000, patience=50, verbose=1, progress_callback=None):
        """
        Train the model.

        Args:
            X_train, y_train: Training set.
            X_val, y_val: Validation set (used for early stopping).
            epochs (int): Maximum number of epochs.
            patience (int): Patience for early stopping.
            verbose (int): 0 - no logs, 1 - with progress bar, 2 - per epoch.
            progress_callback (func): Function to update progress bar.

        Returns:
            history: Training history.

        Raises:
            ValueError: If model is not built (model is None).
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info(f"Starting model training. Epochs: {epochs}, Patience: {patience}")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1
            )
        ]
        if progress_callback:
            callbacks.append(ProgressCallback(epochs, progress_callback))

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val), # Use test as validation for early stopping
            epochs=epochs,
            batch_size=1, # Stochastic learning (r=0.05)
            callbacks=callbacks,
            verbose=verbose
        )
        logger.info("Training completed.")
        return self.history

    def predict(self, X):
        """
        Make a prediction.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predictions.

        Raises:
            ValueError: If model is not trained (model is None).
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, y_test_class=None):
        """
        Evaluate the model on the test set.

        Args:
            X_test, y_test: Test set (for regression).
            y_test_class: Test set (for classification).

        Returns:
            dict: Dictionary with metrics and predictions.

        Raises:
            ValueError: If model is not trained (model is None).
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        y_pred = self.predict(X_test)

        results = {}
        if self.task == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            logger.info(f"Test MSE (norm.): {mse:.6f}, MAE (norm.): {mae:.6f}, RMSE (norm.): {rmse:.6f}")
            results.update({"MSE": mse, "MAE": mae, "RMSE": rmse, "y_pred": y_pred})
        elif self.task == 'classification':
            y_pred_class = np.argmax(y_pred, axis=1) # Get predicted classes
            accuracy = accuracy_score(y_test_class, y_pred_class)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test_class, y_pred_class, average='macro', zero_division=0)
            conf_matrix = confusion_matrix(y_test_class, y_pred_class)
            logger.info(f"Test Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}")
            results.update({"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "Confusion_Matrix": conf_matrix, "y_pred_class": y_pred_class})

        return results

class ProgressCallback(keras.callbacks.Callback):
    """
    Custom callback to update the tkinter progress bar.
    """
    def __init__(self, total_epochs, progress_callback):
        """
        Initialize the callback.

        Args:
            total_epochs (int): Total number of epochs.
            progress_callback (func): Function to update progress bar (takes value 0-100).
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_callback = progress_callback

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.
        Updates the progress bar.
        """
        progress = int(100 * (epoch + 1) / self.total_epochs)
        self.progress_callback(progress)

class FuzzyLogicApp:
    """
    Class for the tkinter GUI.
    """
    def __init__(self, root):
        """
        Initialize the application.

        Args:
            root (tk.Tk): Main tkinter window.
        """
        self.root = root
        self.root.title("Time Series Forecasting - MLP (Variant 15)")

        # --- Load dataset parameters ---
        self.datasets_params_df = self.load_datasets_params()

        # --- Variables to store parameters ---
        self.dataset_path_var = tk.StringVar(value="dataset_15_Sensor_Events.csv")
        self.dataset_list = self.find_dataset_files() # List of dataset files
        self.dataset_combobox_var = tk.StringVar(value="dataset_15_Sensor_Events.csv")
        self.depth_var = tk.IntVar(value=6) # For dataset 15 from datasets_parameters.csv
        self.hidden_neurons_var = tk.IntVar(value=12) # Recommended initial value for dataset 15
        self.activation_var = tk.StringVar(value="tanh")
        self.learning_rate_var = tk.DoubleVar(value=0.05) # From assignment
        self.max_error_var = tk.DoubleVar(value=0.005) # From assignment

        # --- New variables ---
        self.optimizer_var = tk.StringVar(value="SGD")
        self.use_dropout_var = tk.BooleanVar(value=False)
        self.dropout_rate_var = tk.DoubleVar(value=0.1)
        self.use_batch_norm_var = tk.BooleanVar(value=False)
        self.task_var = tk.StringVar(value="regression") # 'regression' or 'classification'
        self.use_tsu_var = tk.BooleanVar(value=HAS_TSU) # Whether to use time_series_utils

        # --- Variables for experiment series ---
        self.hidden_neurons_list_var = tk.StringVar(value="8, 12, 18")
        self.activation_list_var = tk.StringVar(value="tanh, sigmoid")
        self.depth_list_var = tk.StringVar(value="4, 6") # New list for depth
        self.dataset_list_var = tk.StringVar(value="dataset_15_Sensor_Events.csv") # New list for datasets

        # --- Variables to store results ---
        self.current_results = {}
        self.current_history = None
        self.data_processor = None
        self.forecaster = None

        # --- Style ---
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # --- Create menu ---
        self.setup_menu()

        # --- Create tabs ---
        self.setup_ui()

    def load_datasets_params(self):
        """
        Loads datasets_parameters.csv into a DataFrame.

        Returns:
            pd.DataFrame or None: DataFrame with parameters or None if file not found.
        """
        try:
            df = pd.read_csv("datasets_parameters.csv")
            logger.info("datasets_parameters.csv loaded.")
            return df
        except FileNotFoundError:
            logger.warning("datasets_parameters.csv not found. Using default values.")
            return None

    def find_dataset_files(self):
        """
        Finds all dataset_*.csv files in the current directory.

        Returns:
            list: Sorted list of filenames.
        """
        import glob
        files = glob.glob("dataset_*.csv")
        files.sort()
        return files

    def setup_menu(self):
        """
        Creates the application menu.
        """
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Menu "File"
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Dataset...", command=self.browse_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Menu "Settings"
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Save Configuration...", command=self.save_config)
        settings_menu.add_command(label="Load Configuration...", command=self.load_config)

        # Menu "Help"
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def setup_ui(self):
        """
        Creates UI elements using tabs.
        """
        # Create Notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Tab 1: Parameters and Control ---
        tab_params = ttk.Frame(notebook)
        notebook.add(tab_params, text="Parameters and Control")

        # Place parameters and control buttons on this tab
        self.setup_parameters_frame(tab_params)
        self.setup_control_buttons_frame(tab_params)

        # --- Tab 2: Log ---
        tab_log = ttk.Frame(notebook)
        notebook.add(tab_log, text="Log")
        self.setup_log_frame(tab_log)

        # --- Tab 3: Results and Table ---
        tab_results = ttk.Frame(notebook)
        notebook.add(tab_results, text="Results and Table")
        self.setup_results_table_frame(tab_results)

        # --- Tab 4: Plots ---
        tab_plots = ttk.Frame(notebook)
        notebook.add(tab_plots, text="Plots")
        self.setup_plots_frame(tab_plots)

        # --- Tab 5: Dataset Description and Listing ---
        tab_info = ttk.Frame(notebook)
        notebook.add(tab_info, text="Description and Listing")
        self.setup_info_frame(tab_info)

        # --- Tab 6: Forecast ---
        tab_forecast = ttk.Frame(notebook)
        notebook.add(tab_forecast, text="Forecast")
        self.setup_forecast_frame(tab_forecast)

        # Set sizes
        self.root.geometry("1200x800") # Initial size

    def setup_parameters_frame(self, parent):
        """
        Creates the frame with model parameters on the 'Parameters and Control' tab.

        Args:
            parent (tk.Widget): Parent widget (tab).
        """
        params_frame = ttk.LabelFrame(parent, text="Model Parameters")
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        # --- Main parameters ---
        ttk.Label(params_frame, text="Select Dataset:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        dataset_combo = ttk.Combobox(params_frame, textvariable=self.dataset_combobox_var, values=self.dataset_list, state="readonly", width=30)
        dataset_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        dataset_combo.bind("<<ComboboxSelected>>", self.on_dataset_combo_change) # Update path on selection

        ttk.Label(params_frame, text="Embedding Depth (n):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(params_frame, from_=1, to=20, textvariable=self.depth_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(params_frame, text="Number of Hidden Layer Neurons:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(params_frame, from_=1, to=50, textvariable=self.hidden_neurons_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(params_frame, text="Activation Function:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        activation_combo = ttk.Combobox(params_frame, textvariable=self.activation_var, values=["tanh", "sigmoid"], state="readonly", width=8)
        activation_combo.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(params_frame, text="Initial Learning Rate (r):").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(params_frame, textvariable=self.learning_rate_var, width=10).grid(row=4, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(params_frame, text="Max Acceptable Error (ε):").grid(row=5, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(params_frame, textvariable=self.max_error_var, width=10).grid(row=5, column=1, sticky="w", padx=5, pady=2)

        # --- New parameters ---
        ttk.Label(params_frame, text="Optimizer:").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        optimizer_combo = ttk.Combobox(params_frame, textvariable=self.optimizer_var, values=["SGD", "Adam", "RMSprop"], state="readonly", width=8)
        optimizer_combo.grid(row=6, column=1, sticky="w", padx=5, pady=2)

        ttk.Checkbutton(params_frame, text="Use Dropout", variable=self.use_dropout_var).grid(row=7, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(params_frame, textvariable=self.dropout_rate_var, width=10).grid(row=7, column=1, sticky="w", padx=5, pady=2)

        ttk.Checkbutton(params_frame, text="Use BatchNormalization", variable=self.use_batch_norm_var).grid(row=8, column=0, sticky="w", padx=5, pady=2)

        ttk.Label(params_frame, text="Task:").grid(row=9, column=0, sticky="w", padx=5, pady=2)
        task_combo = ttk.Combobox(params_frame, textvariable=self.task_var, values=["regression", "classification"], state="readonly", width=8)
        task_combo.grid(row=9, column=1, sticky="w", padx=5, pady=2)

        ttk.Checkbutton(params_frame, text="Use time_series_utils", variable=self.use_tsu_var).grid(row=10, column=0, sticky="w", padx=5, pady=2)

        # --- Parameters for experiment series ---
        exp_frame = ttk.LabelFrame(parent, text="Parameters for Experiment Series (comma-separated)")
        exp_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(exp_frame, text="List of neuron counts:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(exp_frame, textvariable=self.hidden_neurons_list_var, width=20).grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(exp_frame, text="List of activation functions:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(exp_frame, textvariable=self.activation_list_var, width=20).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(exp_frame, text="List of depths (depth):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(exp_frame, textvariable=self.depth_list_var, width=20).grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(exp_frame, text="List of datasets:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(exp_frame, textvariable=self.dataset_list_var, width=20).grid(row=3, column=1, sticky="w", padx=5, pady=2)

    def setup_control_buttons_frame(self, parent):
        """
        Creates the frame with control buttons on the 'Parameters and Control' tab.

        Args:
            parent (tk.Widget): Parent widget (tab).
        """
        # Create Canvas with horizontal scroll for buttons
        canvas_container = tk.Frame(parent)
        canvas_container.pack(fill=tk.X, padx=10, pady=5)

        button_canvas = tk.Canvas(canvas_container)
        button_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        scrollbar = ttk.Scrollbar(canvas_container, orient="horizontal", command=button_canvas.xview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.X)

        button_canvas.configure(xscrollcommand=scrollbar.set)

        button_inner_frame = tk.Frame(button_canvas)
        button_canvas.create_window((0, 0), window=button_inner_frame, anchor="nw")

        # Configure scrolling
        def on_configure(event):
            button_canvas.configure(scrollregion=button_canvas.bbox("all"))

        button_inner_frame.bind("<Configure>", on_configure)

        # Add buttons to inner_frame
        buttons = [
            ("Load and Process Data", self.load_and_process_data),
            ("Train Model", self.train_model),
            ("Evaluate Model", self.evaluate_model),
            ("Show Plots", self.show_plots_in_new_window),
            ("Save Results", self.save_results),
            ("Export Table", self.export_table),
            ("Show First 10 Windows", self.show_training_samples),
            ("Copy Code Listing", self.copy_code),
            ("Save Model", self.save_trained_model),
            ("Run Experiment Series", self.run_experiment_series),
            ("Save Training History", self.save_training_history),
            ("Load Training History", self.load_training_history),
            ("Forecast Next Step", self.predict_next_step)
        ]

        for text, command in buttons:
            btn = ttk.Button(button_inner_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5, pady=5)

    def setup_log_frame(self, parent):
        """
        Creates the frame for the action log and progress bar.

        Args:
            parent (tk.Widget): Parent widget (tab).
        """
        log_frame = ttk.LabelFrame(parent, text="Action Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Progress bar
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(log_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

    def setup_results_table_frame(self, parent):
        """
        Creates the frame for displaying results and the comparison table.

        Args:
            parent (tk.Widget): Parent widget (tab).
        """
        results_frame = ttk.LabelFrame(parent, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, side=tk.TOP)

        self.results_text = scrolledtext.ScrolledText(results_frame, state='disabled')
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        table_frame = ttk.LabelFrame(parent, text="Comparison Table of Results")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, side=tk.BOTTOM)

        # Create Treeview for table
        # Updated columns
        columns = ("Dataset", "Task", "Activation", "Optimizer", "Hidden Neurons", "Depth", "Uses Dropout", "Dropout Rate", "Uses BatchNorm", "MSE_norm", "MAE_norm", "RMSE_norm", "MSE_actual", "MAE_actual", "RMSE_actual", "Accuracy", "F1-Score")
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        self.results_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5, side=tk.LEFT)

        # Configure headers
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=80, anchor='center') # Reduce width

        # Add scrollbar for Treeview
        tree_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.configure(yscrollcommand=tree_scrollbar.set)

        # Add button to add current results to table
        add_button = ttk.Button(table_frame, text="Add Current Results to Table", command=self.add_to_table)
        add_button.pack(pady=5)

    def setup_plots_frame(self, parent):
        """
        Creates the frame for displaying plots.

        Args:
            parent (tk.Widget): Parent widget (tab).
        """
        plot_frame = ttk.LabelFrame(parent, text="Plots")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create Canvas with Scrollbar
        canvas_container = tk.Frame(plot_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for drawing plots
        self.plot_canvas = tk.Canvas(canvas_container)
        self.plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.plot_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Link Canvas and Scrollbar
        self.plot_canvas.configure(yscrollcommand=scrollbar.set)

        # Frame inside Canvas where plots will be placed
        self.plot_inner_frame = tk.Frame(self.plot_canvas)
        self.plot_canvas.create_window((0, 0), window=self.plot_inner_frame, anchor="nw")

        # Configure scrolling
        def on_configure(event):
            self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))

        self.plot_inner_frame.bind("<Configure>", on_configure)

    def setup_info_frame(self, parent):
        """
        Creates the frame with dataset 15 description, attribute/class buttons, and code listing button.

        Args:
            parent (tk.Widget): Parent widget (tab).
        """
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Dataset description
        desc_frame = ttk.LabelFrame(info_frame, text="Dataset 15 Description")
        desc_frame.pack(fill=tk.X, padx=5, pady=5)

        desc_text = (
            "DATASET 15: SENSOR ALARM EVENTS\n"
            "Category: IoT and security systems\n"
            "Count: 168 observations (hourly over a week)\n"
            "Range: from 1 to 80 events\n"
            "Feature: Daily activity cycle\n"
            "Recommended Number of Neurons: 12-18\n"
            "Application: Anomaly detection in security systems\n"
            "Difficulty Level: Low\n"
        )
        desc_label = tk.Label(desc_frame, text=desc_text, justify=tk.LEFT, anchor='w', wraplength=500)
        desc_label.pack(fill=tk.X, padx=5, pady=5)

        # Buttons for attributes and classes
        attr_button = ttk.Button(info_frame, text="Show Attribute List", command=self.show_attributes)
        attr_button.pack(pady=2)
        class_button = ttk.Button(info_frame, text="Show Class Name List", command=self.show_classes)
        class_button.pack(pady=2)

        # Button to copy code listing
        copy_button = ttk.Button(info_frame, text="Copy Program Code Listing", command=self.copy_code)
        copy_button.pack(pady=10)

    def show_attributes(self):
        """
        Shows a window with information about attribute lists.
        """
        attr_text = (
            "Attribute List:\n"
            "- Embedding Depth (n): 6\n"
            "- Forecasting Horizon (m): 1\n"
            "- Periodicity: hourly\n"
            "- Number of Features (input values): n=6\n"
            "- Target Variable: next value after the window\n"
        )
        messagebox.showinfo("Attribute List", attr_text)

    def show_classes(self):
        """
        Shows a window with information about class name lists.
        """
        class_text = (
            "Class Name List:\n"
            "- Class 1: Low activity (values < 25)\n"
            "- Class 2: Normal activity (25 <= values <= 50)\n"
            "- Class 3: High activity (values > 50)\n"
        )
        messagebox.showinfo("Class Name List", class_text)

    def setup_forecast_frame(self, parent):
        """
        Creates the frame for entering data and forecasting the next step.

        Args:
            parent (tk.Widget): Parent widget (tab).
        """
        forecast_frame = ttk.LabelFrame(parent, text="Forecast Next Step")
        forecast_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(forecast_frame, text="Enter the last n values (comma-separated):").pack(anchor='w', padx=5, pady=5)
        self.forecast_input_var = tk.StringVar()
        ttk.Entry(forecast_frame, textvariable=self.forecast_input_var, width=50).pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(forecast_frame, text="Forecast", command=self.predict_next_step).pack(pady=5)

        self.forecast_result_label = tk.Label(forecast_frame, text="Result will appear here.")
        self.forecast_result_label.pack(pady=5)

    def log_message(self, message):
        """
        Adds a message to the log.

        Args:
            message (str): Message to add.
        """
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def browse_dataset(self):
        """
        Opens a dialog box to select a dataset CSV file.
        """
        filename = filedialog.askopenfilename(
            title="Select Dataset CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.dataset_path_var.set(filename)
            # Update Combobox if file is in the list
            if filename in self.dataset_list:
                self.dataset_combobox_var.set(filename)

    def on_dataset_combo_change(self, event):
        """
        Handler for the dataset combobox selection change event.
        Updates dataset_path_var and substitutes recommended parameters from datasets_parameters.csv.

        Args:
            event: Combobox selection change event.
        """
        selected_file = self.dataset_combobox_var.get()
        self.dataset_path_var.set(selected_file)

        if self.datasets_params_df is not None:
            try:
                # Extract dataset ID from filename
                dataset_id = int(selected_file.split('_')[1])
                row = self.datasets_params_df[self.datasets_params_df['Dataset_ID'] == dataset_id]
                if not row.empty:
                    recommended_depth = row['Depth_n'].iloc[0]
                    recommended_neurons = row['Recommended_Hidden_Neurons'].iloc[0]
                    # Assume the Recommended_Hidden_Neurons column contains a range like "12-18"
                    # Extract the first number as the minimum recommended
                    try:
                        min_neurons = int(recommended_neurons.split('-')[0])
                    except (ValueError, AttributeError):
                        min_neurons = recommended_neurons
                    self.depth_var.set(recommended_depth)
                    self.hidden_neurons_var.set(min_neurons)
                    self.log_message(f"Recommended parameters substituted: Depth={recommended_depth}, Hidden Neurons={min_neurons} for {selected_file}")
            except (ValueError, IndexError):
                # If unable to extract ID, ignore
                pass

    def load_and_process_data(self):
        """
        Loads and processes data from the selected CSV file.
        Performs normalization, creates windows, splits the dataset.
        """
        try:
            path = self.dataset_path_var.get()
            depth = self.depth_var.get()
            activation = self.activation_var.get()
            use_tsu = self.use_tsu_var.get()
            # Choose normalization range based on activation
            scaler_range = (-1, 1) if activation == 'tanh' else (0, 1)

            if not os.path.exists(path):
                messagebox.showerror("Error", f"File not found: {path}")
                return

            self.log_message(f"Loading and processing data from {path}...")
            self.data_processor = TimeSeriesDatasetProcessor(path, depth, scaler_range=scaler_range, use_tsu=use_tsu)
            self.data_processor.load_and_preprocess()

            # Check data for NaN and infinity
            if np.any(np.isnan(self.data_processor.X)) or np.any(np.isinf(self.data_processor.X)):
                messagebox.showwarning("Warning", "Data contains NaN or infinity. Try changing parameters or cleaning data.")
                self.log_message("Warning: Data contains NaN or infinity.")
            elif np.any(np.isnan(self.data_processor.y)) or np.any(np.isinf(self.data_processor.y)):
                messagebox.showwarning("Warning", "Target values contain NaN or infinity. Try changing parameters or cleaning data.")
                self.log_message("Warning: Target values contain NaN or infinity.")

            self.log_message("Data successfully loaded and processed.")
            self.update_results_display(f"Data loaded. Original series length: {len(self.data_processor.data)}, "
                                        f"X_train shape: {self.data_processor.X_train.shape}, "
                                        f"X_test shape: {self.data_processor.X_test.shape}")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.log_message(f"Error: {e}")
            messagebox.showerror("Error", f"Error loading data: {e}")

    def train_model(self):
        """
        Creates and trains the model based on current parameters and loaded data.
        """
        if self.data_processor is None:
            messagebox.showwarning("Warning", "First, load and process data.")
            return

        try:
            self.log_message("Creating model...")
            input_size = self.data_processor.X_train.shape[1] # Embedding depth
            output_size = 1 if self.task_var.get() == 'regression' else 3 # 3 classes for classification
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

            self.log_message("Training model...")
            self.current_history = self.forecaster.train(
                self.data_processor.X_train, self.data_processor.y_train if task == 'regression' else self.data_processor.y_train_class,
                self.data_processor.X_test, self.data_processor.y_test if task == 'regression' else self.data_processor.y_test_class, # Use classes for validation
                progress_callback=self.update_progress
            )
            self.log_message("Training completed.")
            self.progress_var.set(0) # Reset progress

        except Exception as e:
            logger.error(f"Error training model: {e}")
            self.log_message(f"Error: {e}")
            messagebox.showerror("Error", f"Error training model: {e}")

    def update_progress(self, value):
        """
        Updates the progress bar value in the interface.

        Args:
            value (int): Progress value (0-100).
        """
        self.progress_var.set(value)
        self.root.update_idletasks() # Update interface

    def evaluate_model(self):
        """
        Evaluates the trained model on the test set.
        Prints metrics in the results field and updates current_results.
        """
        if self.forecaster is None:
            messagebox.showwarning("Warning", "First, train the model.")
            return

        try:
            self.log_message("Evaluating model on test set...")
            task = self.task_var.get()
            if task == 'regression':
                eval_results = self.forecaster.evaluate(self.data_processor.X_test, self.data_processor.y_test)
                # Revert values to original scale
                y_test_actual = self.data_processor.inverse_transform(self.data_processor.y_test)
                y_pred_actual = self.data_processor.inverse_transform(eval_results['y_pred'])

                # Calculate metrics on original scale
                mse_actual = mean_squared_error(y_test_actual, y_pred_actual)
                mae_actual = mean_absolute_error(y_test_actual, y_pred_actual)
                rmse_actual = np.sqrt(mse_actual)

                # Add original scale metrics to results
                eval_results['MSE_actual'] = mse_actual
                eval_results['MAE_actual'] = mae_actual
                eval_results['RMSE_actual'] = rmse_actual
                eval_results['y_test_actual'] = y_test_actual
                eval_results['y_pred_actual'] = y_pred_actual

                result_str = (
                    f"--- Model Evaluation Results (Regression) ---\n"
                    f"Parameters: Neurons={self.hidden_neurons_var.get()}, Activation={self.activation_var.get()}, Optimizer={self.optimizer_var.get()}\n"
                    f"Test MSE (norm.): {eval_results['MSE']:.6f}\n"
                    f"Test MAE (norm.): {eval_results['MAE']:.6f}\n"
                    f"Test RMSE (norm.): {eval_results['RMSE']:.6f}\n"
                    f"Test MSE (orig.): {eval_results['MSE_actual']:.6f}\n"
                    f"Test MAE (orig.): {eval_results['MAE_actual']:.6f}\n"
                    f"Test RMSE (orig.): {eval_results['RMSE_actual']:.6f}\n"
                )
            elif task == 'classification':
                eval_results = self.forecaster.evaluate(self.data_processor.X_test, None, y_test_class=self.data_processor.y_test_class)
                conf_matrix_str = "\n".join(["\t".join(map(str, row)) for row in eval_results['Confusion_Matrix']])
                result_str = (
                    f"--- Model Evaluation Results (Classification) ---\n"
                    f"Parameters: Neurons={self.hidden_neurons_var.get()}, Activation={self.activation_var.get()}, Optimizer={self.optimizer_var.get()}\n"
                    f"Test Accuracy: {eval_results['Accuracy']:.6f}\n"
                    f"Test Precision (macro): {eval_results['Precision']:.6f}\n"
                    f"Test Recall (macro): {eval_results['Recall']:.6f}\n"
                    f"Test F1-score (macro): {eval_results['F1']:.6f}\n"
                    f"Confusion Matrix:\n{conf_matrix_str}\n"
                )

            # Add dataset 15 description
            description = (
                f"\n--- Dataset Description (Variant 15) ---\n"
                f"Name: Sensor Alarm Events\n"
                f"Category: IoT and security systems\n"
                f"Number of Observations: 168 (hourly over a week)\n"
                f"Value Range: from 1 to 80 events\n"
                f"Feature: Daily activity cycle\n"
                f"Recommended Number of Neurons: 12-18\n"
                f"Application: Anomaly detection in security systems\n"
                f"Difficulty Level: Low\n"
            )
            result_str += description
            self.update_results_display(result_str)

            self.current_results = eval_results  # Store all results

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            self.log_message(f"Error: {e}")
            messagebox.showerror("Error", f"Error evaluating model: {e}")

    def update_results_display(self, text):
        """
        Updates the text field with results.

        Args:
            text (str): Text to display.
        """
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state='disabled')

    def show_plots_in_new_window(self):
        """
        Opens a new window and displays training and prediction plots.
        """
        if not self.current_history or not self.current_results:
            messagebox.showwarning("Warning", "First, train and evaluate the model.")
            return

        # Create new window
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Model Plots")
        plot_window.geometry("1200x800")

        # Create Canvas with Scrollbar for the new window
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

        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Model Results (Neurons: {self.hidden_neurons_var.get()}, Activation: {self.activation_var.get()}, Optimizer: {self.optimizer_var.get()})')

        # 1. Training history (Loss)
        axes[0, 0].plot(self.current_history.history['loss'], label='Training Loss')
        if 'val_loss' in self.current_history.history:
            axes[0, 0].plot(self.current_history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training History (Loss)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. MAE or Accuracy history
        task = self.task_var.get()
        if task == 'regression':
            metric_name = 'mae'
            metric_label = 'MAE'
        else: # classification
            metric_name = 'accuracy'
            metric_label = 'Accuracy'
        axes[0, 1].plot(self.current_history.history[metric_name], label=f'Training {metric_label}')
        if f'val_{metric_name}' in self.current_history.history:
            axes[0, 1].plot(self.current_history.history[f'val_{metric_name}'], label=f'Validation {metric_label}')
        axes[0, 1].set_title(f'{metric_label} History')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel(metric_label)
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. Predictions vs Actual (in original scale, if regression)
        if task == 'regression' and 'y_test_actual' in self.current_results:
            y_test_actual = self.current_results['y_test_actual']
            y_pred_actual = self.current_results['y_pred_actual']
            axes[1, 0].plot(y_test_actual, label='Actual Values', alpha=0.7)
            axes[1, 0].plot(y_pred_actual, label='Predictions', alpha=0.7)
            axes[1, 0].set_title('Predictions vs Actual (Original Scale)')
            axes[1, 0].set_xlabel('Index')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'Plot unavailable for classification task', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Predictions vs Actual (Original Scale)')

        # 4. Residuals (in original scale, if regression)
        if task == 'regression' and 'y_test_actual' in self.current_results:
            residuals = self.current_results['y_test_actual'].flatten() - self.current_results['y_pred_actual'].flatten()
            axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_title('Residuals (Actual - Predicted)')
            axes[1, 1].set_xlabel('Index')
            axes[1, 1].set_ylabel('Residual')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Plot unavailable for classification task', horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Residuals (Actual - Predicted)')

        plt.tight_layout()

        # Embed figure in tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_inner_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Update scrollregion
        plot_canvas.update_idletasks()
        plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))

        # Add "Save Plot" button to the new window
        save_plot_button = ttk.Button(plot_window, text="Save Plot as PNG", command=lambda: self.save_current_plot(fig))
        save_plot_button.pack(pady=5)

    def save_current_plot(self, fig):
        """
        Saves the current matplotlib figure as a PNG file.

        Args:
            fig (matplotlib.figure.Figure): Figure object to save.
        """
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            try:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.log_message(f"Plot saved as {filename}")
                messagebox.showinfo("Save", f"Plot successfully saved as {filename}")
            except Exception as e:
                logger.error(f"Error saving plot: {e}")
                messagebox.showerror("Error", f"Error saving plot: {e}")

    def save_results(self):
        """
        Saves current results (parameters and metrics) to a JSON file.
        """
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to save.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                # Prepare data for saving
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
                self.log_message(f"Results saved to {filename}")
                messagebox.showinfo("Save", f"Results successfully saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                messagebox.showerror("Error", f"Error saving results: {e}")

    def add_to_table(self):
        """
        Adds current parameters and metrics to the TreeView results table.
        """
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to add.")
            return

        try:
            # Get parameters from GUI
            dataset_name = os.path.basename(self.dataset_path_var.get()).replace('.csv', '')
            task = self.task_var.get()
            activation = self.activation_var.get()
            optimizer = self.optimizer_var.get()
            hidden_neurons = self.hidden_neurons_var.get()
            depth = self.depth_var.get()
            uses_dropout = self.use_dropout_var.get()
            dropout_rate = self.dropout_rate_var.get()
            uses_batch_norm = self.use_batch_norm_var.get()

            # Get metrics
            row_data = [dataset_name, task, activation, optimizer, hidden_neurons, depth, uses_dropout, dropout_rate, uses_batch_norm]

            if task == 'regression':
                row_data.extend([
                    f"{self.current_results.get('MSE', 'N/A'):.6f}",
                    f"{self.current_results.get('MAE', 'N/A'):.6f}",
                    f"{self.current_results.get('RMSE', 'N/A'):.6f}",
                    f"{self.current_results.get('MSE_actual', 'N/A'):.6f}",
                    f"{self.current_results.get('MAE_actual', 'N/A'):.6f}",
                    f"{self.current_results.get('RMSE_actual', 'N/A'):.6f}",
                    "N/A", "N/A" # For Accuracy and F1 in regression
                ])
            elif task == 'classification':
                row_data.extend([
                    "N/A", "N/A", "N/A", # For MSE, MAE, RMSE in classification
                    "N/A", "N/A", "N/A", # For MSE_actual, MAE_actual, RMSE_actual in classification
                    f"{self.current_results.get('Accuracy', 'N/A'):.6f}",
                    f"{self.current_results.get('F1', 'N/A'):.6f}"
                ])

            # Add to table
            self.results_tree.insert('', 'end', values=row_data)

            self.log_message(f"Results for {dataset_name} (task={task}) added to table.")

        except Exception as e:
            logger.error(f"Error adding to table: {e}")
            messagebox.showerror("Error", f"Error adding to table: {e}")

    def export_table(self):
        """
        Exports TreeView table contents to a CSV file.
        """
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Headers
                    headers = [self.results_tree.heading(col, option="text") for col in self.results_tree["columns"]]
                    writer.writerow(headers)
                    # Data
                    for item in self.results_tree.get_children():
                        row = self.results_tree.item(item)['values']
                        writer.writerow(row)
                self.log_message(f"Results table saved to {filename}")
                messagebox.showinfo("Export", f"Table successfully saved to {filename}")
            except Exception as e:
                logger.error(f"Error exporting table: {e}")
                messagebox.showerror("Error", f"Error exporting table: {e}")

    def show_training_samples(self):
        """
        Shows the first N examples of the training set (X and y) in the results field.
        N is the minimum of 10 and the size of X_train.
        """
        if self.data_processor is None:
            messagebox.showwarning("Warning", "First, load and process data.")
            return

        try:
            # Take the first min(10, len(X_train)) rows of X_train and y_train
            num_samples = min(10, len(self.data_processor.X_train))
            sample_X_norm = self.data_processor.X_train[:num_samples]
            sample_y_norm = self.data_processor.y_train[:num_samples]

            # Revert to original scale
            sample_X_actual = self.data_processor.inverse_transform(sample_X_norm)
            sample_y_actual = self.data_processor.inverse_transform(sample_y_norm.reshape(-1, 1)).flatten()

            # Form string for output
            sample_str = f"First {num_samples} training examples (normalized):\n"
            sample_str += "X (history) -> y (forecast)\n"
            for i in range(len(sample_X_norm)):
                x_str = ", ".join([f"{x:.6f}" for x in sample_X_norm[i]])
                y_str = f"{sample_y_norm[i]:.6f}"
                sample_str += f"[{x_str}] -> [{y_str}]\n"

            sample_str += f"\nFirst {num_samples} training examples (original scale):\n"
            sample_str += "X (history) -> y (forecast)\n"
            for i in range(len(sample_X_actual)):
                x_str = ", ".join([f"{x:.2f}" for x in sample_X_actual[i]])
                y_str = f"{sample_y_actual[i]:.2f}"
                sample_str += f"[{x_str}] -> [{y_str}]\n"

            self.update_results_display(sample_str)
            self.log_message(f"First {num_samples} training examples shown (norm. and orig. scale).")

        except Exception as e:
            logger.error(f"Error showing samples: {e}")
            messagebox.showerror("Error", f"Error showing samples: {e}")

    def copy_code(self):
        """
        Copies the entire main script source code to the clipboard.
        """
        try:
            # Read the content of the current file
            with open(__file__, 'r', encoding='utf-8') as f:
                code = f.read()
            self.root.clipboard_clear()
            self.root.clipboard_append(code)
            self.log_message("Program listing copied to clipboard.")
            messagebox.showinfo("Copy", "Program listing copied to clipboard.")
        except Exception as e:
            logger.error(f"Error copying code: {e}")
            messagebox.showerror("Error", f"Error copying code: {e}")

    def show_about(self):
        """
        Shows the 'About' window.
        """
        about_text = (
            "Time Series Forecasting Program\n"
            "Laboratory Work #2 on 'AI Fundamentals'\n"
            "Variant 15: Sensor Alarm Events\n"
            "Author: Kolosov Stanislav\n"
            "Date: 2025"
        )
        messagebox.showinfo("About", about_text)

    def save_config(self):
        """
        Saves current interface parameters to a JSON file.
        """
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
                self.log_message(f"Configuration saved to {filename}")
                messagebox.showinfo("Save", f"Configuration successfully saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving configuration: {e}")
                messagebox.showerror("Error", f"Error saving configuration: {e}")

    def load_config(self):
        """
        Loads interface parameters from a JSON file.
        """
        filename = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # Set variable values
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

                self.log_message(f"Configuration loaded from {filename}")
                messagebox.showinfo("Load", f"Configuration successfully loaded from {filename}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                messagebox.showerror("Error", f"Error loading configuration: {e}")

    def run_experiment_series(self):
        """
        Runs a series of experiments with different parameter combinations.
        """
        # if self.data_processor is None: # Remove check, as we will reload data
        #    messagebox.showwarning("Warning", "First, load and process data.")
        #    return

        try:
            # Get parameter lists from GUI
            hidden_neurons_str = self.hidden_neurons_list_var.get()
            activation_str = self.activation_list_var.get()
            depth_str = self.depth_list_var.get()
            dataset_str = self.dataset_list_var.get() # New list

            # Convert strings to lists
            hidden_neurons_list = [int(x.strip()) for x in hidden_neurons_str.split(',') if x.strip().isdigit()]
            activation_list = [x.strip() for x in activation_str.split(',') if x.strip() in ["tanh", "sigmoid"]]
            depth_list = [int(x.strip()) for x in depth_str.split(',') if x.strip().isdigit()]
            dataset_list = [x.strip() for x in dataset_str.split(',') if x.strip() in self.dataset_list] # Validate against list

            if not hidden_neurons_list or not activation_list or not depth_list or not dataset_list: # Add check for dataset_list
                messagebox.showerror("Error", "Parameter lists for experiments are incorrect or empty.")
                return

            self.log_message(f"Running experiment series.")
            self.log_message(f"Neurons: {hidden_neurons_list}, Activations: {activation_list}, Depths: {depth_list}, Datasets: {dataset_list}")

            # Run in a separate thread
            thread = threading.Thread(target=self._run_experiments_thread, args=(hidden_neurons_list, activation_list, depth_list, dataset_list))
            thread.start()

        except Exception as e:
            logger.error(f"Error running experiment series: {e}")
            messagebox.showerror("Error", f"Error running experiment series: {e}")

    def _run_experiments_thread(self, hidden_neurons_list, activation_list, depth_list, dataset_list):
        """
        Internal function to execute experiments in a separate thread.

        Args:
            hidden_neurons_list (list): List of neuron counts.
            activation_list (list): List of activation functions.
            depth_list (list): List of depths.
            dataset_list (list): List of dataset files.
        """
        combinations = list(itertools.product(dataset_list, hidden_neurons_list, activation_list, depth_list))
        total = len(combinations)
        self.log_message(f"Total combinations: {total}")

        for i, (ds_file, hn, act, d) in enumerate(combinations):
            self.log_message(f"Experiment {i+1}/{total}: Dataset={ds_file}, Neurons={hn}, Activation={act}, Depth={d}")

            # Update parameters
            self.dataset_path_var.set(ds_file)
            self.dataset_combobox_var.set(ds_file)
            self.hidden_neurons_var.set(hn)
            self.activation_var.set(act)
            self.depth_var.set(d)
            # Load data
            self.load_and_process_data() # Reload with new parameters
            # Train model
            self.train_model()
            # Evaluate model
            self.evaluate_model()
            # Add results to table
            self.add_to_table()

        self.log_message("Experiment series completed.")

    def save_trained_model(self):
        """
        Saves the trained Keras model to a file.
        """
        if self.forecaster is None or self.forecaster.model is None:
            messagebox.showwarning("Warning", "First, train the model.")
            return

        directory = filedialog.askdirectory(title="Select Folder to Save Model")
        if directory:
            try:
                model_path = os.path.join(directory, "my_trained_model")
                self.forecaster.model.save(model_path)
                self.log_message(f"Model saved to {model_path}")
                messagebox.showinfo("Save", f"Model successfully saved to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                messagebox.showerror("Error", f"Error saving model: {e}")

    def save_training_history(self):
        """
        Saves the training history (history.history) to a JSON file.
        """
        if self.current_history is None:
            messagebox.showwarning("Warning", "No training history to save.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.current_history.history, f, indent=4, ensure_ascii=False)
                self.log_message(f"Training history saved to {filename}")
                messagebox.showinfo("Save", f"Training history successfully saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving history: {e}")
                messagebox.showerror("Error", f"Error saving history: {e}")

    def load_training_history(self):
        """
        Loads training history from a JSON file.
        """
        filename = filedialog.askopenfilename(
            title="Select Training History File",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    history_dict = json.load(f)
                # Create a dummy history object
                class DummyHistory:
                    def __init__(self, hist_dict):
                        self.history = hist_dict
                self.current_history = DummyHistory(history_dict)
                self.log_message(f"Training history loaded from {filename}")
                messagebox.showinfo("Load", f"Training history successfully loaded from {filename}")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                messagebox.showerror("Error", f"Error loading history: {e}")

    def predict_next_step(self):
        """
        Performs a forecast for the next value based on the entered last n values.
        """
        if self.forecaster is None or self.forecaster.model is None:
            messagebox.showwarning("Warning", "First, train the model.")
            return

        try:
            input_str = self.forecast_input_var.get()
            if not input_str:
                messagebox.showwarning("Warning", "Enter values for forecasting.")
                return

            input_values = [float(x.strip()) for x in input_str.split(',')]
            if len(input_values) != self.depth_var.get():
                messagebox.showerror("Error", f"The number of entered values ({len(input_values)}) does not match the embedding depth ({self.depth_var.get()}).")
                return

            # Normalize the entered values using the current scaler
            # This might be inaccurate if the entered values are outside the training data range
            # It's better to use a scaler trained on the general dataset or specify min/max manually
            # For now, use the current scaler
            scaler = self.data_processor.scaler
            normalized_input = scaler.transform(np.array(input_values).reshape(-1, 1)).flatten()
            X_new = normalized_input.reshape(1, -1) # Form batch (1, depth)

            # Prediction
            y_pred_norm = self.forecaster.predict(X_new)
            # Denormalize the result
            y_pred_actual = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()[0]

            # Class
            y_pred_class = self.data_processor.value_to_class(y_pred_actual)

            result_text = f"Forecast (norm.): {y_pred_norm[0][0]:.6f}\nForecast (orig.): {y_pred_actual:.2f}\nClass: {y_pred_class[0]}"
            self.forecast_result_label.config(text=result_text)
            self.log_message(f"Forecast for next step: {y_pred_actual:.2f} (Class: {y_pred_class[0]})")

        except ValueError:
            messagebox.showerror("Error", "Invalid input format. Enter numbers separated by commas.")
        except Exception as e:
            logger.error(f"Error forecasting: {e}")
            messagebox.showerror("Error", f"Error forecasting: {e}")


def main():
    """
    Entry point of the application.
    """
    root = tk.Tk()
    app = FuzzyLogicApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
