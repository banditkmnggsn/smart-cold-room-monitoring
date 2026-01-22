# TinyML Cold Room Monitoring System
## Project Structure & Documentation

---

## 📂 Project Overview

This project implements an **AI-assisted IoT monitoring system for cold rooms** using TinyML.

**Objective**: Detect three conditions:
- **NORMAL** (0): Cold room operating correctly
- **WARM_EXCURSION** (1): Temperature too high (alert!)
- **SENSOR_FAULT** (2): DHT11 sensor malfunction or reading error

**Target Hardware**: ESP32 DevKit V1 with ~12 neurons, 1-2 layers (TensorFlow Lite Micro)

---

## 📁 Folder Structure

```
edgeimpulse-mlp-python/
├── data/
│   ├── Dataset Train (raw).csv              ← Original raw data (173 samples)
│   ├── dataset_engineered_raw.csv           ← Features extracted (173 samples)
│   ├── dataset_engineered_augmented.csv     ← Augmented dataset (300 samples)
│   ├── dataset_for_edgeimpulse.csv          ← Formatted for Edge Impulse
│   └── LABELING_GUIDE.txt                   ← Label definitions
│
├── src/
│   ├── 01_feature_engineering.py            ← Feature extraction & augmentation
│   ├── 02_data_exploration.py               ← Data visualization & analysis
│   ├── 03_prepare_for_edgeimpulse.py        ← Final dataset preparation
│   ├── 04_train_model.py                    ← (Next) Model training
│   └── 05_export_tflite.py                  ← (Next) TFLite export
│
├── model/
│   ├── model.h5                             ← (Generated) Keras model
│   ├── model.tflite                         ← (Generated) TFLite model
│   └── model_quantized.tflite               ← (Generated) Quantized for ESP32
│
├── plots/
│   ├── 01_features_boxplot.png              ← Feature distributions
│   ├── 02_correlation_heatmap.png           ← Feature correlations
│   ├── 03_scatter_plots.png                 ← Pairwise relationships
│   ├── 04_histograms.png                    ← Class distributions
│   └── 05_feature_importance.png            ← Top features
│
├── notebooks/
│   └── (Jupyter notebooks for interactive analysis - optional)
│
├── .venv/                                   ← Virtual environment
├── requirements.txt                         ← Python dependencies
└── README.md                                ← This file
```

---

## 🔧 Installation & Setup

### 1. Create Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 📊 Dataset Pipeline

### Stage 1: Feature Engineering
**Input**: `Dataset Train (raw).csv` (173 samples)

**Script**: `src/01_feature_engineering.py`

**Output**: 
- `dataset_engineered_raw.csv` (173 samples + engineered features)
- `dataset_engineered_augmented.csv` (300 samples with synthetic data)

**Features Created**:
```
Raw Features (4):
- temperature1_c, humidity1_percent
- temperature2_c, humidity2_percent

Engineered Features (11):
- dT1, dT2                          → Temperature delta (rate of change)
- avgT1_5, avgT2_5                  → 5-point rolling average (trend)
- stdT1_5, stdT2_5                  → 5-point std dev (noise detection)
- stuck_count1, stuck_count2        → Sensor frozen detection
- diffT                             → abs(T1 - T2) for offset detection
- abs_dT1, abs_dT2                  → Magnitude of change
```

**Total Features**: 15 inputs + 1 output (label)

### Stage 2: Data Augmentation
**Strategy**: Generate synthetic data for underrepresented classes

- **NORMAL** (100 samples): Stable temp, small dT, low noise
- **WARM_EXCURSION** (100 samples): Rising temp, positive dT trend
- **SENSOR_FAULT** (100 samples): High diffT, stuck values, or spikes

**Result**: Balanced 300-sample dataset

### Stage 3: Data Exploration
**Script**: `src/02_data_exploration.py`

**Outputs**:
- Feature distribution analysis
- Correlation matrix heatmap
- Scatter plots for key feature pairs
- Feature importance ranking

**Top 5 Important Features**:
1. `diffT` (F-ratio: 1609.77) ← Sensor offset detection
2. `dT2` (F-ratio: 239.24)    ← Temp change rate
3. `temperature2_c` (F-ratio: 149.93)
4. `avgT2_5` (F-ratio: 140.02)
5. `stdT1_5` (F-ratio: 73.99)  ← Noise level

### Stage 4: Prepare for Edge Impulse
**Script**: `src/03_prepare_for_edgeimpulse.py`

**Output**: `dataset_for_edgeimpulse.csv`
- Clean CSV format (300 rows × 16 columns)
- Ready to upload to Edge Impulse or train locally

---

## 🎯 Class Definitions

### Class 0: NORMAL (Normal Operation)
**Characteristics**:
- Stable temperature: `dT1, dT2 < 0.1°C`
- Both sensors agree: `diffT < 0.5°C`
- No sensor issues: `stuck_count ≈ 0`, `stdT < 0.3`

**Example**:
```
T1 = 7.1°C, T2 = 7.2°C, diffT = 0.1°C, dT1 = 0.0°C → NORMAL ✓
```

### Class 1: WARM_EXCURSION (Temperature Alert!)
**Characteristics**:
- Higher than normal: `T1 > 10°C` or `T2 > 10°C`
- Rising trend: `dT1 > 0.1°C` or `dT2 > 0.1°C` (consistently positive)
- Sensors still agree: `diffT < 0.5°C`

**Example**:
```
T1 = 10.5°C, T2 = 10.2°C, diffT = 0.3°C, dT1 = +0.3°C → WARM_EXCURSION ⚠️
```

**Action**: Alert staff to check refrigeration unit!

### Class 2: SENSOR_FAULT (Hardware Problem)
**Characteristics** (any of):
- Large offset: `diffT > 2°C` (sensors completely disagree)
- Sensor stuck: `stuck_count > 3` (same reading for 5+ samples)
- Unrealistic spike: `abs_dT > 2°C` (sudden large jump)
- High noise: `stdT > 1.0` (very inconsistent readings)

**Example 1 - Offset**:
```
T1 = 12°C, T2 = 7°C, diffT = 5°C → SENSOR_FAULT 🔴
```

**Example 2 - Stuck**:
```
dT1 = 0.0°C for 6 samples, stuck_count1 = 6 → SENSOR_FAULT 🔴
```

**Example 3 - Spike**:
```
T1: 7.1 → 11.8°C (jump +4.7°C), abs_dT1 = 4.7 → SENSOR_FAULT 🔴
```

**Action**: Replace DHT11 sensor!

---

## 📈 Class Separation Analysis

### Feature Statistics by Class

```
NORMAL:
- Mean Temp1: 7.6°C (±0.9)
- Mean diffT: 0.14°C (±0.09)
- Mean dT1:   0.00°C (±0.03)

WARM_EXCURSION:
- Mean Temp1: 10.1°C (±1.3)
- Mean diffT: 0.25°C (±0.14)
- Mean dT1:   +0.31°C (±0.12)

SENSOR_FAULT:
- Mean Temp1: 9.9°C (±3.9)
- Mean diffT: 3.86°C (±0.90)  ← HUGE difference!
- Mean dT1:   -0.23°C (±2.01) ← High variability!
```

**Key Insight**: 
- `diffT` alone can separate SENSOR_FAULT from others (~1610 F-ratio)
- `dT1/dT2` trends distinguish WARM_EXCURSION
- Combinations needed for optimal accuracy

---

## 🚀 Next Steps

### 1. Model Training (src/04_train_model.py - Next)
```python
# Small MLP for ESP32
Model: Input(15) → Dense(12) → Dense(8) → Dense(3, softmax)
Optimizer: Adam
Loss: Categorical Crossentropy
Epochs: 100-200
Validation: 20% split
```

**Expected Accuracy**: 95-99% (well-separated classes)

### 2. Export to TFLite (src/05_export_tflite.py - Next)
```
model.h5 → model.tflite → model_quantized.tflite (INT8)
Size: ~5-10KB (ESP32 friendly)
```

### 3. Deploy to ESP32
```cpp
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "model_quantized.tflite.h"  // Converted from model

// In ESP32 code:
// 1. Read DHT11 sensors
// 2. Calculate 15 features
// 3. Run TFLite inference
// 4. Get prediction + confidence
// 5. Send to cloud (Blynk/Arduino Cloud/ThingsBoard)
```

---

## 📝 Usage

### Generate all outputs:
```powershell
# Feature engineering & augmentation
python src/01_feature_engineering.py

# Data exploration & visualization
python src/02_data_exploration.py

# Prepare for Edge Impulse
python src/03_prepare_for_edgeimpulse.py
```

### Check results:
- `data/dataset_for_edgeimpulse.csv` - Main training dataset
- `plots/` folder - Visualizations
- `data/LABELING_GUIDE.txt` - Label definitions

---

## 🔍 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 300 |
| Classes | 3 (balanced) |
| Samples per class | 100 |
| Features | 15 engineered |
| Feature types | Numeric (float32) |
| Missing values | 0 |
| Outliers | Intentional (augmented) |

---

## 🎓 Key Concepts

### Why These 15 Features?

1. **Raw sensors** (4): Baseline measurements
2. **Rate of change** (2): `dT` detects trends vs spikes
3. **Moving average** (2): Smooth out noise, see trend
4. **Standard deviation** (2): Quantify noise/instability
5. **Stuck detection** (2): Count static readings
6. **Sensor offset** (1): `diffT` catches mismatches
7. **Absolute deltas** (2): Magnitude for spike detection

### Why 3 Classes?

- **NORMAL** → Keep running, no action
- **WARM_EXCURSION** → HIGH PRIORITY alert
- **SENSOR_FAULT** → Maintenance needed

This triad covers the company's main problems:
✓ Temperature excursions (caught late → now early!)
✓ Sensor faults (DHT stuck/spike → detected!)
✓ No dashboard → Cloud integration coming next

---

## 🛠 Technologies

| Component | Technology |
|-----------|-----------|
| Data processing | Pandas, NumPy |
| ML framework | TensorFlow/Keras |
| Deployment | TensorFlow Lite Micro |
| Hardware | ESP32 DevKit V1 |
| Cloud (optional) | Blynk, Arduino Cloud, or ThingsBoard |
| Development | Python 3.11, VS Code |

---

## 📞 Support

**Questions about labels?** → See `data/LABELING_GUIDE.txt`

**Need to retrain?** → Modify `src/04_train_model.py` (coming next)

**Edge Impulse export?** → Use `data/dataset_for_edgeimpulse.csv`

---

## 📅 Version History

- v1.0 (Current)
  - ✅ Feature engineering complete
  - ✅ Data augmentation (300 samples)
  - ✅ Data exploration & visualization
  - ✅ Dataset ready for Edge Impulse / training
  - ⏳ Model training (next step)
  - ⏳ TFLite export (next step)
  - ⏳ ESP32 deployment (final step)

---

**Last Updated**: January 21, 2026
**Status**: Ready for Model Training 🚀
