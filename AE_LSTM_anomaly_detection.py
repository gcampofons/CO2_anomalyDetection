"""
CO2 Anomaly Detection in IoT Edge Devices using LSTM Autoencoders

This project uses TensorFlow to develop anomaly detection models for CO2 air quality data.
The approach combines Autoencoders with Long Short-Term Memory (LSTM) networks to identify
anomalous patterns in CO2 measurements from IoT sensors in edge computing environments.

Training and evaluation data comes from CEIP Albea Valld'Alba school sensors,
available at Zenodo (https://doi.org/10.5281/zenodo.5036228).

Author: Guillem Campo Fons
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.stats import shapiro

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Dropout, RepeatVector

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    roc_auc_score, recall_score, confusion_matrix
)

from matplotlib import pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration parameters for the CO2 anomaly detection pipeline.

    This dataclass contains all configurable parameters for data processing,
    model training, evaluation, and visualization. Parameters are validated
    in the __post_init__ method.

    Attributes:
        outlier_treatment: Method for handling outliers ('NONE', 'ELIMINATION', 'CAPPING').
        normalization: Data normalization method ('STANDARD', 'MINMAX', 'ROBUST').
        upper_limit: CO2 threshold in PPM for anomaly detection.
        resampling_frequency: Pandas frequency string for data resampling.
        contamination_rate: Expected proportion of anomalies in the data.
        time_steps: Number of time steps in LSTM sequences.
        learning_rate: Learning rate for model training.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        validation_split: Fraction of data for validation.
        early_stopping_patience: Epochs to wait before early stopping.
        loss_function: Loss function for training ('mae', 'mse').
        threshold_loss: Loss type for threshold calculation ('MAE', 'MSE').
        anomaly_loss: Loss type for anomaly detection ('MAE', 'MSE').
        threshold_percentile: Percentile for anomaly threshold calculation.
        plot_fragment_size: Number of sequences to show in reconstruction plots.
        random_seed: Random seed for reproducibility.
        data_file: Path to the input CSV data file.
        output_dir: Directory for saving outputs and models.
    """
    # Data processing
    outlier_treatment: str = 'NONE'  # NONE, ELIMINATION, or CAPPING
    normalization: str = 'ROBUST'  # STANDARD, MINMAX, or ROBUST
    upper_limit: int = 1000  # CO2 threshold for anomaly detection (1000 ppm = recommended indoor limit)
    resampling_frequency: str = '5T'
    contamination_rate: float = 0.05  # Expected % of anomalies in data
    
    # Model training
    time_steps: int = 4
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 64
    validation_split: float = 0.2
    early_stopping_patience: int = 15

    # Loss functions
    loss_function: str = 'mae'
    threshold_loss: str = 'MAE'
    anomaly_loss: str = 'MAE'
    threshold_percentile: float = 90.0
    
    # Visualization
    plot_fragment_size: int = 30
    random_seed: int = 1993
    
    # Paths
    data_file: str = 'CEIP_Albea_ValldAlba.csv'
    output_dir: Path = Path('output')
    
    def __post_init__(self):
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If any configuration parameter has an invalid value.
        """
        valid_treatments = ['NONE', 'ELIMINATION', 'CAPPING']
        valid_normalizations = ['STANDARD', 'MINMAX', 'ROBUST']
        valid_losses = ['mae', 'mse']
        valid_loss_types = ['MAE', 'MSE']
        
        if self.outlier_treatment not in valid_treatments:
            raise ValueError(f"outlier_treatment must be one of {valid_treatments}")
        if self.normalization not in valid_normalizations:
            raise ValueError(f"normalization must be one of {valid_normalizations}")
        if self.loss_function not in valid_losses:
            raise ValueError(f"loss_function must be one of {valid_losses}")
        if self.threshold_loss not in valid_loss_types:
            raise ValueError(f"threshold_loss must be one of {valid_loss_types}")
        if self.anomaly_loss not in valid_loss_types:
            raise ValueError(f"anomaly_loss must be one of {valid_loss_types}")
        
        self.output_dir.mkdir(exist_ok=True)


class DataProcessor:
    """Handles data loading, preprocessing, and transformation."""
    
    def __init__(self, config: Config):
        """Initialize the DataProcessor.

        Args:
            config: Configuration object containing data processing parameters.
        """
        self.config = config
        self.scaler = None
        
    def load_and_preprocess_data(
        self, 
        sensor_ids: List[str] = [' CO2_02']
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess CO2 data from CSV file.
        
        Args:
            sensor_ids: List of sensor IDs to filter
            
        Returns:
            Tuple of (processed_data, uncapped_data)
        """
        logger.info(f"Loading data from {self.config.data_file}")
        
        if not Path(self.config.data_file).exists():
            raise FileNotFoundError(f"Data file not found: {self.config.data_file}")
        
        df = pd.read_csv(
            self.config.data_file, 
            usecols=['date_time', 'co2', 'sensor_id', 'temp', 'hum', 'bat']
        )
        df['date_time'] = pd.to_datetime(df['date_time'])
        df_sorted = df.sort_values(by='date_time')
        
        # Filter and process each sensor
        df_filtered_list = []
        for sensor_id in sensor_ids:
            sensor_df = df_sorted[df_sorted['sensor_id'] == sensor_id].copy()
            if len(sensor_df) == 0:
                logger.warning(f"No data found for sensor {sensor_id}")
                continue
            sensor_df = self._process_sensor_data(sensor_df)
            df_filtered_list.append(sensor_df)
        
        if not df_filtered_list:
            raise ValueError("No data available after filtering")
        
        # Concatenate all sensors
        df_concatenated = pd.concat(df_filtered_list, ignore_index=True)
        df_concatenated = (df_concatenated
                          .interpolate(method='linear')
                          .drop('date_time', axis=1)
                          .reset_index(drop=True))
        
        df_uncapped = df_concatenated.copy()
        logger.info(f"Data loaded successfully. Shape: {df_concatenated.shape}")
        
        return df_concatenated, df_uncapped
    
    def _process_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process individual sensor data for consistency.

        Args:
            df: Raw sensor data DataFrame.

        Returns:
            Processed DataFrame with cleaned timestamps and removed sensor_id.
        """
        df['date_time'] = pd.to_datetime(df['date_time'])
        # Round timestamps to nearest minute to handle second variations
        df['date_time'] = df['date_time'].dt.round('T')
        df = df.drop_duplicates(subset='date_time', keep='first')
        df = df.drop(['sensor_id'], axis=1)
        df = df.set_index('date_time')
        df = df.reset_index()
        return df
    
    def test_normality(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Perform Shapiro-Wilk test for normality on the data.

        Args:
            data: DataFrame containing the data to test.

        Returns:
            Tuple of (test_statistic, p_value) from the Shapiro-Wilk test.
        """
        np.random.seed(self.config.random_seed)
        stat, p_value = shapiro(data)
        alpha = 0.05
        
        logger.info(f"Shapiro-Wilk Test: Statistic={stat:.3f}, p-value={p_value:.3f}")
        if p_value > alpha:
            logger.info("Data appears to be normally distributed (H0 not rejected)")
        else:
            logger.info("Data does not appear to be normally distributed (H0 rejected)")
        
        return stat, p_value
    
    def apply_outlier_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier treatment based on configuration.

        Args:
            df: Input DataFrame with CO2 data.

        Returns:
            DataFrame with outliers treated according to config.outlier_treatment.
        """
        logger.info(f"Applying outlier treatment: {self.config.outlier_treatment}")
        
        if self.config.outlier_treatment == 'NONE':
            df_treated = df.copy()
            outlier_count = (df['co2'] > self.config.upper_limit).sum()
            logger.info(f"Keeping all data (including {outlier_count} values above {self.config.upper_limit} PPM)")
        elif self.config.outlier_treatment == 'ELIMINATION':
            df_treated = df[df['co2'] <= self.config.upper_limit].copy()
            removed_count = len(df) - len(df_treated)
            logger.info(f"Removed {removed_count} outliers above {self.config.upper_limit} PPM")
        elif self.config.outlier_treatment == 'CAPPING':
            df_treated = df.copy()
            df_treated['co2'] = np.where(
                df_treated['co2'] > self.config.upper_limit,
                self.config.upper_limit,
                df_treated['co2']
            )
            capped_count = (df['co2'] > self.config.upper_limit).sum()
            logger.info(f"Capped {capped_count} values to {self.config.upper_limit} PPM")
        else:
            df_treated = df.copy()
        
        return df_treated
    
    def normalize_data(
        self, 
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize data using specified scaler.

        Args:
            train_data: Training data to fit the scaler.
            test_data: Test data to transform.

        Returns:
            Tuple of (normalized_train, normalized_test) as numpy arrays.
        """
        logger.info(f"Normalizing data using {self.config.normalization} scaler")
        
        scaler_map = {
            'STANDARD': StandardScaler,
            'MINMAX': MinMaxScaler,
            'ROBUST': RobustScaler
        }
        
        self.scaler = scaler_map[self.config.normalization]()
        train_scaled = self.scaler.fit_transform(train_data)
        test_scaled = self.scaler.transform(test_data)
        
        return train_scaled, test_scaled
    
    @staticmethod
    def create_sequences(values: np.ndarray, time_steps: int) -> np.ndarray:
        """Create sequences for LSTM input from time series data.

        Args:
            values: Input data array of shape (n_samples, n_features).
            time_steps: Number of time steps in each sequence.

        Returns:
            Array of sequences with shape (n_sequences, time_steps, n_features).
        """
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i:(i + time_steps)])
        return np.stack(output)
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i:(i + time_steps)])
        return np.stack(output)


class ModelBuilder:
    """Builds and manages LSTM autoencoder models."""
    
    def __init__(self, config: Config):
        """Initialize the ModelBuilder.

        Args:
            config: Configuration object containing model parameters.
        """
        self.config = config
        
    def build_models(self, input_shape: Tuple[int, int]) -> List[Sequential]:
        """Build multiple LSTM autoencoder architectures.

        Creates 6 different model variants: Simple AE-LSTM, Single LSTM,
        Double LSTM, Deep LSTM, CNN-LSTM Hybrid, and Bidirectional LSTM.

        Args:
            input_shape: Shape of input sequences (time_steps, n_features).

        Returns:
            List of compiled Keras Sequential models.
        """
        logger.info("Building LSTM autoencoder models")
        
        models = [
            self._build_model_1(input_shape),
            self._build_model_2(input_shape),
            self._build_model_3(input_shape),
            self._build_model_4(input_shape),
            self._build_model_5(input_shape),
            self._build_model_6(input_shape)
        ]
        
        # Compile all models
        for i, model in enumerate(models, 1):
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss=self.config.loss_function
            )
            logger.info(f"Model {i}: {model.count_params():,} parameters")
        
        return models
    
    def _build_model_1(self, input_shape: Tuple[int, int]) -> Sequential:
        """Simple LSTM Autoencoder with RepeatVector."""
        model = Sequential([
            LSTM(16, activation='tanh', input_shape=input_shape, return_sequences=False),
            RepeatVector(input_shape[0]),
            LSTM(16, activation='tanh', return_sequences=True),
            TimeDistributed(Dense(input_shape[1]))
        ], name='AE_LSTM_Simple')
        return model
    
    def _build_model_2(self, input_shape: Tuple[int, int]) -> Sequential:
        """Single LSTM layer with Dropout."""
        model = Sequential([
            LSTM(16, activation='tanh', input_shape=input_shape, return_sequences=True),
            Dropout(rate=0.2),
            TimeDistributed(Dense(input_shape[1]))
        ], name='AE_LSTM_Single')
        return model
    
    def _build_model_3(self, input_shape: Tuple[int, int]) -> Sequential:
        """Two-layer LSTM with Dropout."""
        model = Sequential([
            LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=True),
            Dropout(rate=0.2),
            LSTM(16, activation='tanh', return_sequences=True),
            TimeDistributed(Dense(input_shape[1]))
        ], name='AE_LSTM_Double')
        return model
    
    def _build_model_4(self, input_shape: Tuple[int, int]) -> Sequential:
        """Three-layer deep LSTM with Dropout."""
        model = Sequential([
            LSTM(128, activation='tanh', input_shape=input_shape, return_sequences=True),
            Dropout(rate=0.2),
            LSTM(64, activation='tanh', return_sequences=True),
            Dropout(rate=0.2),
            LSTM(16, activation='tanh', return_sequences=True),
            TimeDistributed(Dense(input_shape[1]))
        ], name='AE_LSTM_Deep')
        return model
    
    def _build_model_5(self, input_shape: Tuple[int, int]) -> Sequential:
        """CNN-LSTM hybrid for spatial-temporal feature extraction."""
        model = Sequential([
            tf.keras.layers.Conv1D(32, kernel_size=2, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            LSTM(64, activation='tanh', return_sequences=True),
            Dropout(rate=0.2),
            LSTM(16, activation='tanh', return_sequences=True),
            TimeDistributed(Dense(input_shape[1]))
        ], name='CNN_LSTM_Hybrid')
        return model
    
    def _build_model_6(self, input_shape: Tuple[int, int]) -> Sequential:
        """Bidirectional LSTM for better temporal context."""
        model = Sequential([
            tf.keras.layers.Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=input_shape),
            Dropout(rate=0.2),
            tf.keras.layers.Bidirectional(LSTM(32, activation='tanh', return_sequences=True)),
            Dropout(rate=0.2),
            TimeDistributed(Dense(input_shape[1]))
        ], name='Bidirectional_LSTM')
        return model


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: Config):
        """Initialize the ModelTrainer.

        Args:
            config: Configuration object containing training parameters.
        """
        self.config = config
        
    def train_models(
        self, 
        models: List[Sequential], 
        train_data: np.ndarray
    ) -> List[Dict]:
        """
        Train all models.
        
        Args:
            models: List of Keras models
            train_data: Training sequences
            
        Returns:
            List of training histories
        """
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True
        )
        
        histories = []
        for i, model in enumerate(models, 1):
            logger.info(f"Training Model {i}/{len(models)}")
            history = model.fit(
                train_data, train_data,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            logger.info(f"Model {i} training completed. Final loss: {history.history['loss'][-1]:.4f}")
            histories.append(history.history)
        
        return histories
    
    def predict_sequences(
        self, 
        models: List[Sequential], 
        data: np.ndarray
    ) -> List[np.ndarray]:
        """Generate predictions for all models."""
        logger.info("Generating predictions")
        return [model.predict(data, verbose=0) for model in models]
    
    def calculate_loss(
        self, 
        original: np.ndarray, 
        reconstructed: np.ndarray, 
        loss_type: str
    ) -> np.ndarray:
        """Calculate reconstruction loss."""
        if loss_type == 'MAE':
            return np.mean(np.abs(reconstructed - original), axis=(1, 2))
        elif loss_type == 'MSE':
            return np.mean(np.power(original - reconstructed, 2), axis=(1, 2))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def calculate_thresholds(
        self, 
        train_sequences: np.ndarray, 
        train_reconstructed: List[np.ndarray]
    ) -> List[float]:
        """Calculate anomaly detection thresholds."""
        logger.info(f"Calculating thresholds at {self.config.threshold_percentile}th percentile")
        
        thresholds = []
        for i, reconstructed in enumerate(train_reconstructed, 1):
            loss = self.calculate_loss(
                train_sequences, 
                reconstructed, 
                self.config.threshold_loss
            )
            threshold = np.percentile(loss, self.config.threshold_percentile)
            thresholds.append(threshold)
            logger.info(f"Model {i} threshold: {threshold:.6f}")
        
        return thresholds
    
    def detect_anomalies(
        self, 
        test_sequences: np.ndarray, 
        test_reconstructed: List[np.ndarray], 
        thresholds: List[float]
    ) -> List[np.ndarray]:
        """Detect anomalies based on reconstruction error."""
        logger.info("Detecting anomalies")
        
        anomalies_list = []
        for reconstructed, threshold in zip(test_reconstructed, thresholds):
            loss = self.calculate_loss(
                test_sequences, 
                reconstructed, 
                self.config.anomaly_loss
            )
            anomalies = loss.reshape(-1) > threshold
            anomalies_list.append(anomalies)
            logger.info(f"Detected {anomalies.sum()} anomalies")
        
        return anomalies_list


class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, config: Config):
        """Initialize the Visualizer.

        Args:
            config: Configuration object containing visualization parameters.
        """
        self.config = config
        sns.set_style('whitegrid')
        
    def plot_time_series(self, data: pd.DataFrame, title: str, save_name: Optional[str] = None):
        """Plot CO2 time series."""
        fig, ax = plt.subplots(figsize=(18, 5))
        sns.lineplot(x=data.index, y=data['co2'], ax=ax, linewidth=2)
        ax.set_xlabel('Time Order', fontsize=12)
        ax.set_ylabel('CO2 (PPM)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_facecolor('#F5F5F5')
        ax.grid(visible=True, linestyle='solid', alpha=0.5)
        ax.margins(x=0.02, y=0.1)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.config.output_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot: {save_name}.png")
        # plt.show()  # Removed to prevent blocking execution
        
    def plot_training_history(self, histories: List[Dict], save_name: Optional[str] = None):
        """Plot training and validation loss for all models."""
        n_models = len(histories)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 3*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (ax, history) in enumerate(zip(axes, histories), 1):
            ax.plot(history['loss'], label='Training loss', linewidth=2)
            ax.plot(history['val_loss'], label='Validation loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title(f'Model {i} Training History', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        if save_name:
            plt.savefig(self.config.output_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        # plt.show()  # Removed to prevent blocking execution
        
    def plot_reconstruction_comparison(
        self, 
        original: np.ndarray, 
        reconstructed_list: List[np.ndarray],
        save_name: Optional[str] = None
    ):
        """Compare original and reconstructed sequences."""
        fragment = self.config.plot_fragment_size
        all_original = np.concatenate([original[i] for i in range(min(fragment, len(original)))])
        
        n_models = len(reconstructed_list)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        axes = axes.ravel() if n_models > 1 else [axes]
        
        for i, reconstructed in enumerate(reconstructed_list):
            all_reconstructed = np.concatenate([reconstructed[j] for j in range(min(fragment, len(reconstructed)))])
            ax = axes[i]
            ax.plot(all_original, label='Original', linewidth=2, alpha=0.7)
            ax.plot(all_reconstructed, label='Reconstructed', linewidth=2, alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Normalized Value')
            ax.set_title(f'Model {i+1}: Original vs Reconstructed')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        if save_name:
            plt.savefig(self.config.output_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        # plt.show()  # Removed to prevent blocking execution
        
    def plot_anomalies(
        self, 
        data: pd.DataFrame, 
        anomalies_list: List[np.ndarray],
        labels: Optional[List[str]] = None,
        save_name: Optional[str] = None
    ):
        """Plot detected anomalies."""
        n_plots = len(anomalies_list)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        co2_values = data['co2'].values
        
        for i, (ax, anomalies) in enumerate(zip(axes, anomalies_list)):
            ax.plot(co2_values, label='CO2 Values', linewidth=1, alpha=0.7)
            # Adjust indices for sequence-based detection (mark last point of sequence)
            sequence_indices = np.where(anomalies)[0]
            anomaly_indices = sequence_indices + self.config.time_steps - 1
            # Ensure indices are within bounds
            anomaly_indices = anomaly_indices[anomaly_indices < len(co2_values)]
            if len(anomaly_indices) > 0:
                ax.scatter(
                    anomaly_indices, 
                    co2_values[anomaly_indices], 
                    color='red', 
                    label=f'Anomalies ({len(anomaly_indices)})',
                    s=30,
                    zorder=5
                )
            ax.set_xlabel('Time Step')
            ax.set_ylabel('CO2 (PPM)')
            title = labels[i] if labels else f'Anomaly Detection - Model {i}'
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        if save_name:
            plt.savefig(self.config.output_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot: {save_name}.png")
        # plt.show()  # Removed to prevent blocking execution
        
    def plot_confusion_matrices(
        self, 
        confusion_matrices: List[np.ndarray],
        save_name: Optional[str] = None
    ):
        """Plot confusion matrices for all models."""
        n_matrices = len(confusion_matrices)
        n_cols = 2
        n_rows = (n_matrices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
        axes = axes.ravel() if n_matrices > 1 else [axes]
        
        labels = ['Normal', 'Anomalía']
        
        for i, (ax, cm) in enumerate(zip(axes, confusion_matrices)):
            df_cm = pd.DataFrame(cm, index=labels, columns=labels)
            sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
            ax.set_title(f'Confusion Matrix - Model {i+1}', fontsize=12)
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('True Class')
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        if save_name:
            plt.savefig(self.config.output_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        # plt.show()  # Removed to prevent blocking execution


class MetricsCalculator:
    """Calculate and display performance metrics."""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        logger.info(f"{model_name} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray, 
        y_pred_list: List[np.ndarray]
    ) -> pd.DataFrame:
        """Compute metrics for all models and return as DataFrame."""
        results = []
        
        for i, y_pred in enumerate(y_pred_list, 1):
            metrics = MetricsCalculator.compute_metrics(y_true, y_pred, f"Model {i}")
            metrics['model'] = f'Model_{i}'
            results.append(metrics)
        
        df_metrics = pd.DataFrame(results)
        df_metrics = df_metrics[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']]
        
        return df_metrics
    
    @staticmethod
    def compute_confusion_matrices(
        y_true: np.ndarray, 
        y_pred_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Compute confusion matrices for all models."""
        matrices = []
        
        for i, y_pred in enumerate(y_pred_list, 1):
            cm = confusion_matrix(y_true.astype(int), y_pred.astype(int))
            matrices.append(cm)
            logger.info(f"Model {i} Confusion Matrix:\n{cm}")
        
        return matrices


class ModelExporter:
    """Export models to various formats."""
    
    def __init__(self, config: Config):
        """Initialize the ModelExporter.

        Args:
            config: Configuration object containing export parameters.
        """
        self.config = config
        
    def save_models(
        self, 
        models: List[Sequential], 
        save_h5: bool = True, 
        save_tflite: bool = True
    ) -> List[Path]:
        """Save models in H5 and/or TFLite format."""
        logger.info("Saving models")
        saved_files = []
        
        for i, model in enumerate(models, 1):
            if save_h5:
                h5_path = self.config.output_dir / f'model_{i}.h5'
                model.save(h5_path)
                saved_files.append(h5_path)
                logger.info(f"Saved {h5_path.name}")
            
            if save_tflite:
                tflite_path = self.config.output_dir / f'model_{i}.tflite'
                self._convert_to_tflite(model, tflite_path)
                saved_files.append(tflite_path)
                
                # Log file size
                size_kb = tflite_path.stat().st_size / 1024
                logger.info(f"Saved {tflite_path.name} ({size_kb:.2f} KB)")
        
        return saved_files
    
    @staticmethod
    def _convert_to_tflite(model: Sequential, output_path: Path):
        """Convert Keras model to TFLite format."""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, 
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        
        output_path.write_bytes(tflite_model)
    
    def test_tflite_model(
        self, 
        model_path: Path, 
        test_data: np.ndarray, 
        threshold: float,
        indices: List[int] = None
    ):
        """Test TFLite model inference."""
        logger.info(f"Testing TFLite model: {model_path.name}")
        
        # Try to load Flex delegate for LSTM support
        delegates = []
        try:
            flex_delegate = tf.lite.experimental.load_delegate('flex')
            delegates.append(flex_delegate)
            logger.info("Flex delegate loaded successfully")
        except Exception as e:
            logger.warning(f"Flex delegate not available: {e}. TFLite testing may fail.")
        
        interpreter = tf.lite.Interpreter(model_path=str(model_path), experimental_delegates=delegates)
        try:
            interpreter.allocate_tensors()
        except RuntimeError as e:
            if "Flex" in str(e) or "not supported" in str(e):
                logger.warning(f"TFLite model requires Flex delegate which is not available on this platform. Skipping TFLite testing.")
                return
            else:
                raise
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        indices = indices or [0, 1]
        
        for idx in indices:
            if idx >= len(test_data):
                continue
                
            input_data = test_data[idx:idx+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Calculate anomaly
            test_loss = np.mean(np.power(input_data - output_data, 2), axis=1).reshape(-1)
            is_anomaly = test_loss > threshold
            
            logger.info(f"Sample {idx}: Loss={test_loss[0]:.6f}, Anomaly={is_anomaly[0]}")


def identify_ground_truth_anomalies(
    sequences: np.ndarray, 
    upper_limit: float,
    config: Config
) -> np.ndarray:
    """Identify ground truth anomalies based on multiple criteria.

    This function marks sequences as anomalous if they contain extreme CO2 values
    or sudden jumps that exceed statistical thresholds.

    Args:
        sequences: Array of CO2 sequences, shape (n_sequences, seq_length, n_features).
        upper_limit: Upper threshold for extreme CO2 values in PPM.
        config: Configuration object containing anomaly detection parameters.

    Returns:
        Boolean array indicating which sequences are anomalous, shape (n_sequences,).

    Note:
        Uses 99th percentile of absolute changes as dynamic jump threshold.
        Abnormal rate criterion is disabled as it was too sensitive.
    """
    anomalies = np.zeros(len(sequences), dtype=bool)
    
    # Calculate dynamic threshold for sudden jumps based on dataset statistics
    all_changes = []
    for seq in sequences:
        changes = np.abs(np.diff(seq.flatten()))
        all_changes.extend(changes)
    all_changes = np.array(all_changes)
    jump_threshold = np.percentile(all_changes, 99)  # 99th percentile of changes
    logger.info(f"Dynamic jump threshold (99th percentile): {jump_threshold:.2f} PPM")
    
    for i, seq in enumerate(sequences):
        # Criterion 1: Extreme values
        has_extreme = np.any(seq > upper_limit)
        
        # Criterion 2: Sudden jumps (change > dynamic threshold)
        changes = np.abs(np.diff(seq.flatten()))
        has_sudden_jump = np.any(changes > jump_threshold)
        
        # Criterion 3: Abnormal rate of change (removed, as it marks too many sequences as anomalies)
        has_abnormal_rate = False
        
        # Mark as anomaly if any criterion is met
        anomalies[i] = has_extreme or has_sudden_jump or has_abnormal_rate
    
    logger.info(f"Ground truth anomalies: {anomalies.sum()} / {len(anomalies)} ({100*anomalies.sum()/len(anomalies):.1f}%)")
    logger.info(f"  - Extreme values: {sum(np.any(seq > upper_limit) for seq in sequences)}")
    logger.info(f"  - Sudden jumps: {sum(np.any(np.abs(np.diff(seq.flatten())) > jump_threshold) for seq in sequences)}")
    
    return anomalies


def main():
    """Main execution pipeline for CO2 anomaly detection.

    Orchestrates the complete workflow from data loading to model export:
    1. Load and preprocess CO2 sensor data
    2. Build and train multiple LSTM architectures
    3. Evaluate models and detect anomalies
    4. Export trained models for deployment

    This function serves as the entry point when running the script directly.
    """
    # Initialize configuration
    config = Config()
    
    logger.info("=" * 60)
    logger.info("CO2 Anomaly Detection with AE-LSTM")
    logger.info("=" * 60)
    
    # Initialize components
    data_processor = DataProcessor(config)
    model_builder = ModelBuilder(config)
    trainer = ModelTrainer(config)
    visualizer = Visualizer(config)
    exporter = ModelExporter(config)
    
    # 1. Load and preprocess data
    logger.info("STEP 1: Data Loading and Preprocessing")
    logger.info("="*60)
    
    df_concatenated, df_uncapped = data_processor.load_and_preprocess_data()
    
    # Visualize raw data
    visualizer.plot_time_series(
        df_concatenated, 
        'CO2 Recordings Over Time (Raw Data)',
        'co2_raw_data'
    )
    
    # Test normality
    data_processor.test_normality(df_concatenated)
    
    # Apply outlier treatment
    df_treated = data_processor.apply_outlier_treatment(df_concatenated)
    
    # Visualize treated data
    visualizer.plot_time_series(
        df_treated,
        f'CO2 Recordings After {config.outlier_treatment}',
        'co2_treated_data'
    )
    
    # 2. Split and Normalize data
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Data Split and Normalization")
    logger.info("="*60)

    # Use same data for train and test (temporal anomaly detection)
    train_scaled, test_scaled = data_processor.normalize_data(df_treated, df_treated)
    logger.info("Training with all available data (including potential anomalies)")
    logger.info("Model will learn to reconstruct normal patterns - anomalies will have high reconstruction error")
    
    # 3. Create sequences
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Sequence Creation")
    logger.info("="*60)
    
    train_sequences = DataProcessor.create_sequences(train_scaled, config.time_steps)
    test_sequences = DataProcessor.create_sequences(test_scaled, config.time_steps)
    logger.info(f"Train sequences shape: {train_sequences.shape}")
    logger.info(f"Test sequences shape: {test_sequences.shape}")
    
    # 4. Build models
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Model Building")
    logger.info("="*60)
    
    input_shape = (train_sequences.shape[1], train_sequences.shape[2])
    models = model_builder.build_models(input_shape)
    
    # 5. Train models
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Model Training")
    logger.info("="*60)
    
    histories = trainer.train_models(models, train_sequences)
    visualizer.plot_training_history(histories, 'training_history')
    
    # 6. Generate predictions
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Prediction Generation")
    logger.info("="*60)
    
    train_reconstructed = trainer.predict_sequences(models, train_sequences)
    test_reconstructed = trainer.predict_sequences(models, test_sequences)
    
    # Visualize reconstruction
    visualizer.plot_reconstruction_comparison(
        train_sequences,
        train_reconstructed,
        'reconstruction_comparison'
    )
    
    # 7. Calculate thresholds
    logger.info("\n" + "="*60)
    logger.info("STEP 7: Threshold Calculation")
    logger.info("="*60)
    
    thresholds = trainer.calculate_thresholds(train_sequences, train_reconstructed)
    
    # 8. Detect anomalies
    logger.info("\n" + "="*60)
    logger.info("STEP 8: Anomaly Detection")
    logger.info("="*60)
    
    predicted_anomalies = trainer.detect_anomalies(
        test_sequences,
        test_reconstructed,
        thresholds
    )
    
    # Identify ground truth
    test_descaled = data_processor.scaler.inverse_transform(test_scaled)
    test_descaled_df = pd.DataFrame(test_descaled, columns=['co2', 'temp', 'hum', 'bat'])
    test_descaled_sequences = DataProcessor.create_sequences(test_descaled_df, config.time_steps)
    ground_truth_anomalies = identify_ground_truth_anomalies(
        test_descaled_sequences,
        config.upper_limit,
        config
    )
    
    # Visualize anomalies
    all_anomalies = [ground_truth_anomalies] + predicted_anomalies
    labels = ['Ground Truth'] + [f'Model {i}' for i in range(1, len(predicted_anomalies)+1)]
    visualizer.plot_anomalies(
        test_descaled_df,
        all_anomalies,
        labels,
        'anomaly_detection'
    )
    
    # 9. Evaluate models
    logger.info("\n" + "="*60)
    logger.info("STEP 9: Model Evaluation")
    logger.info("="*60)
    
    metrics_df = MetricsCalculator.compute_all_metrics(
        ground_truth_anomalies,
        predicted_anomalies
    )
    
    logger.info("\nMetrics Summary:")
    logger.info("\n" + metrics_df.to_string(index=False))
    
    # Save metrics
    metrics_path = config.output_dir / 'metrics_results.csv'
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\nMetrics saved to {metrics_path}")
    
    # Confusion matrices
    confusion_matrices = MetricsCalculator.compute_confusion_matrices(
        ground_truth_anomalies,
        predicted_anomalies
    )
    visualizer.plot_confusion_matrices(confusion_matrices, 'confusion_matrices')
    
    # 10. Export models
    logger.info("\n" + "="*60)
    logger.info("STEP 10: Model Export")
    logger.info("="*60)
    
    saved_files = exporter.save_models(models, save_h5=True, save_tflite=True)
    
    # Test TFLite model (Model 3 - best performing)
    tflite_model_path = config.output_dir / 'model_3.tflite'
    if tflite_model_path.exists():
        exporter.test_tflite_model(
            tflite_model_path,
            test_sequences,
            thresholds[2],  # Index 2 for Model 3
            indices=[2709, 2711] if len(test_sequences) > 2711 else [0, 1]
        )
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
