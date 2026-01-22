"""
Feature Engineering & Data Augmentation
Untuk TinyML Cold Room Monitoring System
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Get base directory (src folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

RAW_DATA_PATH = os.path.join(DATA_DIR, "Dataset Train (raw).csv")
ENGINEERED_DATA_PATH = os.path.join(DATA_DIR, "dataset_engineered_raw.csv")
OUTPUT_DATA_PATH = os.path.join(DATA_DIR, "dataset_engineered_augmented.csv")

WINDOW_SIZE = 5  # Untuk rolling avg dan std
AUGMENTED_SAMPLES = 300  # Target total samples

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_features(df):
    """
    Calculate engineered features dari raw sensor data.
    
    Features:
    - dT1, dT2: delta temperature (perubahan antar pembacaan)
    - avgT1_5, avgT2_5: rolling average 5 sampel
    - stdT1_5, stdT2_5: rolling std dev 5 sampel
    - stuck_count1, stuck_count2: jumlah pembacaan stabil (<0.1°C)
    - diffT: abs(T1 - T2) untuk deteksi sensor fault
    """
    
    df_feat = df.copy()
    
    # 1. Delta temperature (perubahan dari pembacaan sebelumnya)
    df_feat['dT1'] = df_feat['temperature1_c'].diff().fillna(0)
    df_feat['dT2'] = df_feat['temperature2_c'].diff().fillna(0)
    
    # 2. Rolling average (5 data terakhir)
    df_feat['avgT1_5'] = df_feat['temperature1_c'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
    df_feat['avgT2_5'] = df_feat['temperature2_c'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
    
    # 3. Rolling std dev (5 data terakhir)
    df_feat['stdT1_5'] = df_feat['temperature1_c'].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
    df_feat['stdT2_5'] = df_feat['temperature2_c'].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
    
    # 4. Stuck count (sensor tidak berubah atau perubahan <0.1°C)
    df_feat['stuck_count1'] = (abs(df_feat['dT1']) < 0.1).astype(int)
    df_feat['stuck_count2'] = (abs(df_feat['dT2']) < 0.1).astype(int)
    
    # 5. Temperature difference (deteksi sensor fault)
    df_feat['diffT'] = abs(df_feat['temperature1_c'] - df_feat['temperature2_c'])
    
    # 6. Absolute delta untuk deteksi spike
    df_feat['abs_dT1'] = abs(df_feat['dT1'])
    df_feat['abs_dT2'] = abs(df_feat['dT2'])
    
    return df_feat

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_dataset(df_original, target_samples=AUGMENTED_SAMPLES):
    """
    Augment dataset dengan 3 skenario:
    1. NORMAL (label=0): Suhu stabil, kedua sensor konsisten
    2. WARM_EXCURSION (label=1): Suhu naik perlahan atau tiba-tiba
    3. SENSOR_FAULT (label=2): Sensor stuck atau beda jauh
    """
    
    augmented_data = []
    
    # Jumlah sampel per kelas
    samples_per_class = target_samples // 3
    remainder = target_samples % 3
    
    # ========== SKENARIO 1: NORMAL ==========
    print("Generating NORMAL samples...")
    normal_samples = df_original[df_original['status'] == 'OK'].copy()
    
    for _ in range(samples_per_class + remainder):
        # Ambil row random
        row = normal_samples.sample(1).iloc[0].copy()
        
        # Slight noise untuk avgT (±0.2°C)
        row['avgT1_5'] += np.random.normal(0, 0.1)
        row['avgT2_5'] += np.random.normal(0, 0.1)
        
        # Sensor stabil (dT kecil)
        row['dT1'] = np.random.uniform(-0.05, 0.05)
        row['dT2'] = np.random.uniform(-0.05, 0.05)
        
        # diffT harus kecil (<0.5°C)
        row['diffT'] = np.random.uniform(0, 0.3)
        
        # Low stuck count (sensor berjalan normal)
        row['stuck_count1'] = 0
        row['stuck_count2'] = 0
        
        row['label'] = 0  # NORMAL
        augmented_data.append(row)
    
    # ========== SKENARIO 2: WARM_EXCURSION ==========
    print("Generating WARM_EXCURSION samples...")
    
    for _ in range(samples_per_class):
        row = normal_samples.sample(1).iloc[0].copy()
        
        # Trend: suhu naik perlahan
        row['temperature1_c'] += np.random.uniform(1.0, 3.5)  # Naik 1-3.5°C
        row['temperature2_c'] += np.random.uniform(0.8, 3.0)
        
        row['avgT1_5'] = row['temperature1_c'] + np.random.normal(0, 0.1)
        row['avgT2_5'] = row['temperature2_c'] + np.random.normal(0, 0.1)
        
        # dT positif (naik)
        row['dT1'] = np.random.uniform(0.1, 0.5)
        row['dT2'] = np.random.uniform(0.1, 0.5)
        
        # Masih konsisten antar sensor
        row['diffT'] = np.random.uniform(0, 0.5)
        
        row['stuck_count1'] = 0
        row['stuck_count2'] = 0
        
        row['label'] = 1  # WARM_EXCURSION
        augmented_data.append(row)
    
    # ========== SKENARIO 3: SENSOR_FAULT ==========
    print("Generating SENSOR_FAULT samples...")
    
    for _ in range(samples_per_class):
        row = normal_samples.sample(1).iloc[0].copy()
        
        # Sub-skenario: Sensor fault bisa terjadi dengan beberapa cara
        fault_type = np.random.choice(['stuck', 'spike', 'offset'])
        
        if fault_type == 'stuck':
            # Sensor stuck: nilai tidak berubah
            row['dT1'] = np.random.uniform(-0.01, 0.01)
            row['dT2'] = np.random.uniform(-0.01, 0.01)
            row['stuck_count1'] = np.random.randint(3, 8)
            row['stuck_count2'] = np.random.randint(3, 8)
            
            # Tapi sensor tetap terbaca, cuma beda jauh dari sensor lain
            row['temperature1_c'] += np.random.uniform(2, 5)
            row['diffT'] = np.random.uniform(2.0, 5.0)
            
        elif fault_type == 'spike':
            # Spike: perubahan tiba-tiba & unrealistic
            row['temperature1_c'] += np.random.choice([-5, -4, -3, 3, 4, 5])
            row['dT1'] = np.random.uniform(-5, 5)  # Lompat drastis
            row['abs_dT1'] = abs(row['dT1'])
            row['stdT1_5'] = np.random.uniform(1.5, 3.0)  # High variance
            
            row['diffT'] = np.random.uniform(2.5, 5.5)
            row['stuck_count1'] = 0
            row['stuck_count2'] = 0
            
        else:  # offset
            # Sensor offset: beda jarak konsisten tapi tidak masuk akal
            row['temperature1_c'] += np.random.uniform(3, 6)
            row['temperature2_c'] += np.random.uniform(0.2, 0.5)
            row['diffT'] = np.random.uniform(2.5, 5.5)
            row['avgT1_5'] = row['temperature1_c']
            row['avgT2_5'] = row['temperature2_c']
            row['stuck_count1'] = 0
            row['stuck_count2'] = 0
        
        row['label'] = 2  # SENSOR_FAULT
        augmented_data.append(row)
    
    # Convert to DataFrame
    df_augmented = pd.DataFrame(augmented_data)
    
    # Reset index
    df_augmented = df_augmented.reset_index(drop=True)
    
    return df_augmented

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("FEATURE ENGINEERING & DATA AUGMENTATION")
    print("=" * 70)
    
    # 1. Load raw dataset
    print(f"\n1. Loading raw dataset from {RAW_DATA_PATH}...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"   Loaded {len(df_raw)} rows")
    print(f"   Columns: {list(df_raw.columns)}")
    
    # 2. Feature engineering
    print(f"\n2. Applying feature engineering...")
    df_engineered_raw = calculate_features(df_raw)
    
    # Select features untuk modeling
    feature_cols = [
        'temperature1_c', 'humidity1_percent', 'temperature2_c', 'humidity2_percent',
        'dT1', 'dT2', 'avgT1_5', 'avgT2_5', 'stdT1_5', 'stdT2_5',
        'stuck_count1', 'stuck_count2', 'diffT', 'abs_dT1', 'abs_dT2'
    ]
    
    df_engineered_raw = df_engineered_raw[['index', 'timestamp_ms', 'status'] + feature_cols]
    
    # Save engineered raw
    df_engineered_raw.to_csv(ENGINEERED_DATA_PATH, index=False)
    print(f"   ✓ Saved engineered features to {ENGINEERED_DATA_PATH}")
    print(f"   Shape: {df_engineered_raw.shape}")
    
    # 3. Data augmentation
    print(f"\n3. Augmenting dataset to {AUGMENTED_SAMPLES} samples...")
    df_augmented = augment_dataset(df_engineered_raw, target_samples=AUGMENTED_SAMPLES)
    
    # 4. Save augmented dataset
    df_augmented.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"   ✓ Saved augmented dataset to {OUTPUT_DATA_PATH}")
    print(f"   Shape: {df_augmented.shape}")
    
    # 5. Summary statistics
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"\nOriginal dataset: {len(df_raw)} samples")
    print(f"Engineered dataset: {len(df_engineered_raw)} samples")
    print(f"Augmented dataset: {len(df_augmented)} samples")
    
    print(f"\nLabel distribution (augmented):")
    label_dist = df_augmented['label'].value_counts().sort_index()
    label_map = {0: 'NORMAL', 1: 'WARM_EXCURSION', 2: 'SENSOR_FAULT'}
    for label, count in label_dist.items():
        print(f"  {label_map[label]}: {count} samples ({count/len(df_augmented)*100:.1f}%)")
    
    print(f"\nFeature statistics (augmented):")
    print(df_augmented[feature_cols].describe())
    
    print("\n" + "=" * 70)
    print("✓ Feature engineering and augmentation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
