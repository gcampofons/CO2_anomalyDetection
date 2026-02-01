# CO2 Anomaly Detection with LSTM Models

Temporal anomaly detection for air quality in IoT Edge devices using LSTM-based architectures (including autoencoders and deep LSTMs).

## 📋 Description

This project uses TensorFlow to develop anomaly detection models for CO2 data. It implements 6 different LSTM-based architectures (including autoencoders, deep LSTMs, CNN-LSTM hybrids, and bidirectional LSTMs) to identify temporal anomalous patterns in air quality measurements from IoT sensors.

The model detects:
- ✅ Extreme values (>1000 PPM - exceeds indoor air quality standards)
- ✅ Sudden jumps (dynamic threshold based on 99th percentile of dataset changes)
- ✅ Temporal patterns that deviate from normal behavior

**Author:** Guillem Campo Fons

## 🎯 Model Performance

### Best Model: **Model 1 (Simple AE-LSTM)**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 92.7% | Correctly classifies 93% of sequences |
| **Precision** | 39.0% | 39% of predicted anomalies are real |
| **Recall** | 76.5% | Detects 77% of all real anomalies |
| **F1-Score** | 51.7% | Good balance between precision and recall |
| **AUC-ROC** | 0.851 | Excellent discriminative power |

### Model Comparison (Optimized Configuration)

| Model | Architecture | Accuracy | Recall | F1-Score | AUC |
|-------|-------------|----------|--------|----------|-----|
| **Model 1** ⭐ | Simple AE-LSTM | **92.5%** | **76.3%** | **49.3%** | **0.848** |
| **Model 2** | Single LSTM | 89.7% | 46.7% | 30.2% | 0.693 |
| **Model 3** | Double LSTM | 89.7% | 46.4% | 30.0% | 0.691 |
| **Model 4** | Deep LSTM | 89.7% | 47.1% | 30.4% | 0.695 |
| **Model 5** | CNN-LSTM Hybrid | 89.4% | 44.0% | 28.4% | 0.678 |
| **Model 6** | Bidirectional LSTM | 89.9% | 48.8% | 31.5% | 0.704 |

**Recommendation:** Use **Model 1** for production deployment:
- Best overall performance with realistic ground truth (4.8% anomalies)
- Excellent recall (76.3%) for anomaly detection
- Outstanding AUC (0.848) for pattern discrimination
- Simple and efficient architecture for edge deployment

## � Dataset

**Source**: CEIP Albea Valld'Alba school sensors  
**DOI**: [10.5281/zenodo.5036228](https://doi.org/10.5281/zenodo.5036228)  
**Size**: ~6,000 readings with 5-minute intervals  
**Features**: CO2 (PPM), Temperature (°C), Humidity (%), Battery level  

### Data Acquisition
```bash
# Download the dataset from Zenodo
# Place CEIP_Albea_ValldAlba.csv in the project root directory
wget https://zenodo.org/record/5036228/files/CEIP_Albea_ValldAlba.csv
```

**Note**: The CSV file is not included in this repository due to size considerations. Please download it manually.

## �🚀 Key Features

- **Temporal Pattern Recognition**: Learns normal CO2 behavior patterns over time
- **Multiple Anomaly Criteria**: Detects various types of anomalies beyond simple thresholds
- **Edge-Ready Models**: Exports to TFLite for deployment on IoT devices
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Automated Pipeline**: End-to-end processing from data loading to model export

## 🗂️ Project Structure

```
CO2_anomalyDetection/
├── AE_LSTM_anomaly_detection.py    # Main script
├── CEIP_Albea_ValldAlba.csv        # Dataset (download separately)
├── requirements.txt                 # Project dependencies
├── README.md                        # This file
├── setup.bat                        # Installation script (Windows)
├── setup.sh                         # Installation script (Linux/Mac)
└── output/                          # Results (created automatically)
    ├── *.png                        # Generated plots
    ├── *.h5                         # Keras models
    ├── *.tflite                     # TFLite models
    └── metrics_results.csv          # Evaluation metrics
```

## 🚀 Installation and Setup

### Prerequisites

- Python 3.13 or higher
- pip (Python package manager)

### Option 1: Automatic Installation (Recommended)

**On Windows:**
```powershell
# Double-click setup.bat or run in PowerShell/CMD
setup.bat
```

**On Linux/Mac:**
```bash
# Grant execution permissions
chmod +x setup.sh

# Run script
./setup.sh
```

### Option 2: Manual Installation

#### Step 1: Create Virtual Environment

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get script execution error, run first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**On Windows (CMD):**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat
```

**On Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
# Update pip (recommended)
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### Step 3: Download the Dataset

Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.5036228) and place the `CEIP_Albea_ValldAlba.csv` file in the project root folder.

## ▶️ Usage

### Basic Execution

```bash
# Make sure virtual environment is activated
python AE_LSTM_anomaly_detection.py
```

The script will automatically:
1. Load and preprocess data
2. Train 4 different LSTM-based models
3. Generate predictions and detect anomalies
4. Save visualizations, models, and metrics

**Note:** The entire codebase is in English for international accessibility.

### Configuration Parameters

The script uses a `Config` class for parameter management. Key parameters:

```python
@dataclass
class Config:
    # Data processing
    outlier_treatment: str = 'NONE'  # Keep all data for temporal learning
    normalization: str = 'ROBUST'    # 'STANDARD', 'MINMAX', 'ROBUST'
    upper_limit: int = 968           # For ground truth definition
    contamination_rate: float = 0.05
    
    # Model training - OPTIMIZED
    time_steps: int = 4               # 20-minute windows for better context
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 64              # Better gradient estimation
    early_stopping_patience: int = 15
    
    # Anomaly detection - OPTIMIZED
    threshold_percentile: float = 90.0  # More sensitive detection
    threshold_loss: str = 'MAE'       # More robust threshold calculation
    anomaly_loss: str = 'MAE'         # Consistent with threshold_loss
```

**Key Parameter Notes:**
- `threshold_percentile=90.0`: More sensitive detection (vs 92.0 previously)
- `time_steps=4`: 20-minute windows capture longer temporal patterns
- `batch_size=64`: Better gradient estimation and faster training
- `MAE loss`: More robust to outliers than MSE for threshold calculation

## 📊 Results and Outputs

The pipeline automatically generates:

### 1. Visualizations (saved in `output/`)
- **CO2 Time Series**: Raw and processed data visualization
- **Training History**: Loss curves for all models
- **Reconstruction Comparison**: Original vs reconstructed sequences
- **Anomaly Detection**: Detected anomalies highlighted in red
- **Confusion Matrices**: True/false positives and negatives

### 2. Trained Models
- **Keras format** (.h5): For Python/TensorFlow deployment
- **TFLite format** (.tflite): For Edge/IoT devices (optimized size)

### 3. Evaluation Metrics (CSV)
Detailed performance metrics for all 6 models:
- Accuracy, Precision, Recall
- F1-Score, AUC-ROC

### Sample Results

**Model 1 Performance Analysis:**
```
✅ High Recall (76.3%): Detects majority of anomalies
✅ Good Precision (36.4%): Balanced false positive rate
✅ Excellent AUC (0.848): Strong pattern discrimination
✅ Simple Architecture: Efficient for IoT deployment
```

## 📈 Model Visualizations

### Data Exploration
![CO2 Raw Data](output/co2_raw_data.png)
*Figure 1: Raw CO2 sensor readings over time, showing natural fluctuations and potential anomalies.*

![CO2 Treated Data](output/co2_treated_data.png)
*Figure 2: Processed data after outlier treatment, ready for model training.*

### Training Performance
![Training History](output/training_history.png)
*Figure 3: Training and validation loss curves for all 6 models, showing convergence and potential overfitting.*

### Model Reconstruction Quality
![Reconstruction Comparison](output/reconstruction_comparison.png)
*Figure 4: Comparison between original and reconstructed sequences, demonstrating how well each model learns normal patterns.*

### Anomaly Detection Results
![Anomaly Detection](output/anomaly_detection.png)
*Figure 5: Detected anomalies overlaid on CO2 time series. Red markers indicate sequences flagged as anomalous by each model.*

### Confusion Matrices
![Confusion Matrices](output/confusion_matrices.png)
*Figure 6: Confusion matrices for all models, showing true positives, false positives, true negatives, and false negatives.*

### CO2 Level Standards & Health Guidelines

Based on established air quality standards, CO2 levels are categorized as follows:

| CO2 Level (PPM) | Category | Health Impact | Action Required |
|----------------|----------|---------------|----------------|
| < 350 | Excellent | Optimal air quality | None |
| 350 - 500 | Good | Normal indoor levels | None |
| 500 - 800 | Acceptable | Slightly elevated | Monitor |
| 800 - 1000 | Moderate | Noticeable effects | Improve ventilation |
| 1000 - 1200 | Poor | Headaches, fatigue | Ventilate immediately |
| > 1200 | Dangerous | Severe health risks | Evacuate/ventilate urgently |

**Key Thresholds Used:**
- **1000 ppm**: Maximum recommended for indoor environments (offices, schools)
- **1200 ppm**: Dangerous level requiring immediate action
- **Dynamic jumps**: 99th percentile of changes in dataset (data-driven threshold)

### Model 1: Simple AE-LSTM ⭐ (RECOMMENDED)
```
LSTM(16) → RepeatVector → LSTM(16) → TimeDistributed(Dense(1))
```
- **True autoencoder** with symmetric encoder-decoder structure
- Classic bottleneck compression approach
- **Best performance** (76% recall, 49% F1-score)
- Recommended for production deployment

### Model 2: Single LSTM
```
LSTM(16) → Dropout(0.2) → TimeDistributed(Dense(1))
```
- Lightweight and efficient
- Good performance (43% recall, 29% F1-score)
- Suitable for resource-constrained devices

### Model 3: Double LSTM
```
LSTM(64) → Dropout(0.2) → LSTM(16) → Dense(1)
```
- Two-layer architecture
- Good performance (44% recall, 30% F1-score)

### Model 4: Deep LSTM
```
LSTM(128) → Dropout → LSTM(64) → Dropout → LSTM(16) → TimeDistributed(Dense(1))
```
- Three-layer deep LSTM architecture (not a true autoencoder)
- Good performance (45% recall, 31% F1-score)
- Captures complex temporal dependencies

### Model 5: CNN-LSTM Hybrid
```
Conv1D(32) → MaxPool1D → LSTM(64) → Dropout → LSTM(16) → TimeDistributed(Dense(1))
```
- Combines convolutional feature extraction with LSTM temporal modeling
- Good performance (46% recall, 31% F1-score)
- Effective for pattern recognition in sequential data

### Model 6: Bidirectional LSTM
```
Bidirectional(LSTM(64)) → Dropout → Bidirectional(LSTM(16)) → TimeDistributed(Dense(1))
```
- Processes sequences in both forward and backward directions
- Strong performance (70% recall, 47% F1-score)
- Better context understanding for anomaly detection

## 🧠 How It Works

### Temporal Anomaly Detection Approach

1. **Data Preparation**:
   - Loads CO2 sensor data with 5-minute intervals
   - Applies temporal rounding to handle timestamp variations
   - Keeps all data (including outliers) for comprehensive pattern learning

2. **Training Phase**:
   - **Model learns from ALL data** (normal + anomalous patterns)
   - Uses LSTM-based models (autoencoders and deep LSTMs) to reconstruct/predict temporal sequences
   - Learns to compress and reconstruct 4-step CO2 sequences (20 minutes)
   - Calculates reconstruction/prediction error for each sequence during training

3. **Threshold Calculation**:
   - Computes reconstruction errors on training data
   - Sets anomaly threshold at 92nd percentile of training errors
   - Higher threshold = fewer false positives, lower threshold = fewer false negatives

4. **Detection Phase**:
   - Applies trained model to new sequences
   - Flags sequences with reconstruction error > threshold as anomalies
   - Uses MAE loss for both threshold calculation and anomaly detection

5. **Ground Truth Definition** (Health-based criteria):
   - **Extreme values**: Any reading >**1000 ppm** (exceeds recommended indoor limit)
   - **Sudden jumps**: Changes >99th percentile of all changes in dataset
   - **Result**: 4.8% of sequences marked as anomalous (realistic distribution)

### Key Innovation

Unlike simple threshold-based detection, this approach:
- **Learns temporal context**: Understands normal CO2 fluctuation patterns
- **Detects contextual anomalies**: Normal values can be anomalous in unusual sequences
- **Handles multiple anomaly types**: Beyond just high values
- **Adapts to data distribution**: Learns from actual sensor behavior patterns

## 🔧 Troubleshooting

### Error: ModuleNotFoundError

**Solution:** Ensure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Error: FileNotFoundError (dataset)

**Solution:** Download the dataset and ensure `CEIP_Albea_ValldAlba.csv` is in the project root folder.

### Error: GPU not available

**Solution:** The code works on CPU. For GPU acceleration, install:
```bash
pip install tensorflow[and-cuda]
```

### TensorFlow Warnings (oneDNN, AVX-512)

These are informational messages and don't affect execution. They indicate CPU optimizations are not available but the code will use default implementations.

### Low Recall / High False Negatives

**Solution:** Decrease `threshold_percentile` (e.g., from 92 to 90) in Config class.

### Too Many False Positives

**Solution:** Increase `threshold_percentile` (e.g., from 92 to 95) in Config class.

## 💡 Optimization Tips

### For Better Recall (Detect More Anomalies)
- Decrease `threshold_percentile` (90-92) → More sensitive detection
- Increase `contamination_rate` → Assumes more anomalies in data
- Use `anomaly_loss='MAE'` instead of `'MSE'` → Different error calculation
- Increase `time_steps` (4-5) → Longer temporal context

### For Better Precision (Fewer False Alarms)
- Increase `threshold_percentile` (94-96) → More conservative detection
- Decrease `contamination_rate` → Assumes fewer anomalies
- Add more training `epochs` (150-200) → Better pattern learning
- Use `threshold_loss='MAE'` → Different threshold calculation

### For Faster Training
- Reduce `epochs` (50-75) → Faster convergence
- Increase `batch_size` (64) → More efficient GPU utilization
- Use only Model 2 → Skip training other architectures
- Reduce `early_stopping_patience` (10) → Stop earlier

### For Edge Deployment
- Use Model 4 (best performance) or Model 2 (lightest)
- TFLite models are already optimized for mobile/embedded
- Typical inference time: <10ms per sequence on mobile CPU
- Model size: ~50KB (TFLite compressed)

### For Different Datasets
- Adjust `upper_limit` based on your sensor's normal range
- Modify jump threshold (currently 100 PPM) for your data scale
- Tune `rate_of_change` threshold (currently 0.25) for your temporal patterns
- Consider `time_steps` based on your anomaly duration (4 = 20 minutes)

## 📦 Main Dependencies

- **TensorFlow** 2.20+: Deep learning framework with LSTM support
- **Keras** (included in TensorFlow): High-level neural network API
- **NumPy** 2.2.1+: Array operations and mathematical functions
- **Pandas** 2.2.3+: Data manipulation and time series handling
- **Scikit-learn** 1.6.1+: Data preprocessing (StandardScaler, RobustScaler)
- **Matplotlib** 3.10.0+: Plotting and visualization
- **Seaborn** 0.13.2+: Statistical data visualization
- **SciPy** 1.15.1+: Statistical tests (Shapiro-Wilk normality test)

### Hardware Requirements

- **Minimum**: CPU with AVX support, 4GB RAM
- **Recommended**: GPU with CUDA support, 8GB+ RAM
- **Edge Deployment**: ARM64 CPU, 1GB RAM (TFLite models)

### Python Version
- **Supported**: Python 3.13+
- **Recommended**: Python 3.13+

## ⚠️ Limitations & Future Improvements

### Current Limitations
- **Fixed sequence length**: Uses 4 time steps (20 minutes) - may miss longer anomalies
- **Single sensor focus**: Currently processes one sensor at a time
- **Memory intensive**: Large datasets require significant RAM for sequence creation
- **Hyperparameter sensitivity**: Performance depends on threshold tuning
- **No online learning**: Models must be retrained for concept drift

### Potential Enhancements
- **Multi-sensor fusion**: Combine data from multiple CO2 sensors
- **Variable sequence lengths**: Use attention mechanisms for flexible windows
- **Online adaptation**: Implement incremental learning for concept drift
- **Ensemble methods**: Combine multiple anomaly detection approaches
- **Real-time optimization**: Streaming inference for continuous monitoring
- **Explainability**: Add SHAP values or attention maps for anomaly explanations

### Research Applications
- **IoT Edge computing**: Deploy on resource-constrained devices
- **Smart buildings**: HVAC optimization and air quality monitoring
- **Industrial safety**: Gas leak detection in manufacturing
- **Environmental monitoring**: Pollution pattern analysis
- **Healthcare**: Ventilation system anomaly detection

## 📝 Notes

- Training takes 5-15 minutes depending on hardware
- Models are automatically saved after training
- All visualizations are saved in high-resolution PNG format
- The script uses logging to show progress for each step
- Results are reproducible with `random_seed=1993`

## 📚 References

**Dataset:** CEIP Albea Valld'Alba CO2 Measurements  
[Zenodo DOI: 10.5281/zenodo.5036228](https://doi.org/10.5281/zenodo.5036228)

**Related Work:**
- LSTM Autoencoders for anomaly detection in time series
- Edge computing for IoT air quality monitoring
- Temporal pattern recognition in sensor data

---

## 📚 References & Citation

### Dataset
**CEIP Albea Valld'Alba CO2 Measurements**  
DOI: [10.5281/zenodo.5036228](https://doi.org/10.5281/zenodo.5036228)  
Source: Zenodo Research Data Repository

### Related Research
- LSTM Autoencoders for Time Series Anomaly Detection
- Edge Computing for IoT Applications
- Temporal Pattern Recognition in Sensor Data
- Air Quality Monitoring Systems

### Citation
If you use this code in your research, please cite:

```bibtex
@software{campo_fons_2026_co2_anomaly,
  author       = {Campo Fons, Guillem},
  title        = {CO2 Anomaly Detection with AE-LSTM},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/gcampofons/CO2_anomalyDetection},
  doi          = {10.5281/zenodo.your-doi}
}
```

**Version:** 2.4 
**Last Updated:** February 2026  
**License:** MIT

