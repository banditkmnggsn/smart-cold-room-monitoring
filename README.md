# TinyML Cold Room Monitoring System

## Project Overview

IoT monitoring system for cold rooms using TinyML on ESP32.

**Classes:**
- Class 0: Normal operation (274 samples)
- Class 1: Cooling failure (216 samples)
- Class 2: Sensor fault (210 samples)

**Total Dataset:** 700 samples
**Features:** 15 engineered features
**Target Hardware:** ESP32 with TensorFlow Lite

## Folder Structure

```
edgeimpulse-mlp-python/
├── data/
│   ├── Dataset Train (raw).csv              (Original raw data)
│   ├── dataset_engineered_augmented.csv     (700 samples after feature engineering)
│   ├── dataset_for_edgeimpulse.csv          (Edge Impulse format - ready to upload)
│   └── LABELING_GUIDE.txt
│
├── src/
│   ├── 01_feature_engineering_v2.py
│   ├── 02_export_edgeimpulse.py
│   └── 02_data_exploration.py
│
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### 1. Feature Engineering & Data Augmentation

```powershell
python src/01_feature_engineering_v2.py
```

Output: `data/dataset_engineered_augmented.csv` (700 samples)

### 2. Export for Edge Impulse

```powershell
python src/02_export_edgeimpulse.py
```

Output: `data/dataset_for_edgeimpulse.csv`

### 3. Data Exploration

```powershell
python src/02_data_exploration.py
```

## Dataset Pipeline

### Input
Original dataset: 173 samples from Dataset Train (raw).csv

### Processing
Script: `src/01_feature_engineering_v2.py`

**Features engineered (15 total):**
- Raw: temperature1_c, humidity1_percent, temperature2_c, humidity2_percent
- Delta: dT1, dT2, abs_dT1, abs_dT2
- Trend: avgT1_5, avgT2_5
- Noise: stdT1_5, stdT2_5
- Health: stuck_count1, stuck_count2, diffT

**Data Augmentation:**
- Class 0 (Normal): 274 samples (39.1%)
- Class 1 (Cooling Failure): 216 samples (30.9%)
- Class 2 (Sensor Fault): 210 samples (30.0%)
- Total: 700 samples

### Output Files
- `dataset_engineered_augmented.csv` - 700 samples with 15 features
- `dataset_for_edgeimpulse.csv` - Ready for Edge Impulse upload (timestamp + features + label)

## Class Definitions

**Class 0: Normal Operation**
- Stable temperature (6-8.5°C)
- Low delta (dT < 0.2°C)
- Sensors agree (diffT < 0.3°C)

**Class 1: Cooling Failure**
- High temperature (T > 8.5°C)
- Rising trend (dT > +0.2°C)
- Sensors agree (diffT < 3.5°C)

**Class 2: Sensor Fault**
- Large sensor offset (diffT > 3.5°C)
- Temperature out of range (T < -15°C or T > 50°C)
- Stuck sensor (repeated values)
- High noise (stdT > 1.0)

## Detection Thresholds

- Normal range: 6-8.5°C
- Cooling failure: T > 8.5°C AND dT > 0.2°C
- Sensor fault: diffT > 3.5°C OR T outside [-15, 50]°C

## Dataset Statistics

- Total samples: 700
- Classes: 3 (balanced)
- Features: 15 engineered
- Class 0: 274 samples (39.1%)
- Class 1: 216 samples (30.9%)
- Class 2: 210 samples (30.0%)

## Technologies

- Data processing: Pandas, NumPy
- ML framework: TensorFlow Lite Micro
- Hardware: ESP32 DevKit V1
- Development: Python 3.11
