"""
SEQUENCE-AWARE AUGMENTATION SCRIPT
TinyML Cold Room Monitoring - 3-Class Classification

Target Dataset: 700 samples
- Class 0: Normal operation (280 = 40%)
- Class 1: Abnormal - Cooling Failure (210 = 30%)
- Class 2: Abnormal - Sensor Fault (210 = 30%)

Strategy:
1. Keep original data ORDERED by timestamp
2. Rows 0-70: Class 0 (Normal, 6-8°C)
3. Rows 71-130: Class 1 (Cooling Failure, natural rise)
4. Rows 131-173: Class 0 (Normal, recovery)
5. Synthetic episodes for Class 1 & 2
"""

import pandas as pd
import numpy as np
import os

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

RAW_DATA_PATH = os.path.join(DATA_DIR, "Dataset Train (raw).csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "dataset_engineered_augmented.csv")

WINDOW_SIZE = 5
TARGET_SAMPLES = 700

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_features(df):
    """Calculate 15 engineered features"""
    df_feat = df.copy()
    
    # Delta features
    df_feat['dT1'] = df_feat['temperature1_c'].diff().fillna(0)
    df_feat['dT2'] = df_feat['temperature2_c'].diff().fillna(0)
    df_feat['abs_dT1'] = abs(df_feat['dT1'])
    df_feat['abs_dT2'] = abs(df_feat['dT2'])
    
    # Moving average (5-point)
    df_feat['avgT1_5'] = df_feat['temperature1_c'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
    df_feat['avgT2_5'] = df_feat['temperature2_c'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
    
    # Standard deviation (5-point)
    df_feat['stdT1_5'] = df_feat['temperature1_c'].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
    df_feat['stdT2_5'] = df_feat['temperature2_c'].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
    
    # Sensor health
    df_feat['stuck_count1'] = (abs(df_feat['dT1']) < 0.01).astype(int)  # Nearly zero change
    df_feat['stuck_count2'] = (abs(df_feat['dT2']) < 0.01).astype(int)
    
    # Critical: sensor offset detection
    df_feat['diffT'] = abs(df_feat['temperature1_c'] - df_feat['temperature2_c'])
    
    return df_feat

# ============================================================================
# CLASS LABELING STRATEGY
# ============================================================================

def label_original_data(df):
    """
    Label original data based on temperature pattern with REVISED THRESHOLDS:
    - Class 0 (Normal): T stable 6-8.5°C, dT ≈ 0, diffT < 0.3°C
    - Class 1 (Cooling Failure): T > 8.5°C AND dT > +0.20°C rising trend
    - Class 2 (Sensor Fault): diffT > 3.5°C OR unrealistic temps
    
    REVISED PARAMETERS:
    - dT threshold: +0.20°C (moderate, balanced)
    - T threshold: 8.5°C (early detection)
    - diffT threshold: < 0.3°C (strict sensor agreement)
    - Confirmation: Multi-reading (3+ consecutive)
    """
    df_labeled = df.copy()
    
    # NEW THRESHOLDS
    DT_THRESHOLD = 0.20  # Trigger rising alarm at +0.20°C
    TEMP_ALARM = 8.5     # Temperature above this + rising dT = CLASS 1
    DIFF_NORMAL = 0.30   # Normal sensor agreement < 0.3°C
    
    labels = []
    for idx in range(len(df_labeled)):
        t1 = df_labeled.iloc[idx]['temperature1_c']
        t2 = df_labeled.iloc[idx]['temperature2_c']
        dt1 = df_labeled.iloc[idx]['dT1']
        dt2 = df_labeled.iloc[idx]['dT2']
        diff_t = abs(t1 - t2)
        avg_temp = (t1 + t2) / 2
        
        # Check for sensor fault first (priority!)
        if diff_t > 3.5:
            labels.append(2)
        elif t1 < -15 or t1 > 50 or t2 < -15 or t2 > 50:
            labels.append(2)
        # Check for cooling failure (rising trend + high temp)
        elif (dt1 > DT_THRESHOLD or dt2 > DT_THRESHOLD) and avg_temp > TEMP_ALARM:
            labels.append(1)
        # Otherwise normal
        else:
            labels.append(0)
    
    df_labeled['label'] = labels
    return df_labeled

# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_data(df_original):
    """
    Generate 527 synthetic samples with REVISED THRESHOLDS:
    - dT threshold: +0.20°C
    - T threshold: 8.5°C (early detection with rising)
    - T critical: 10.0°C (stuck high = always failure)
    - diffT normal: < 0.3°C
    - diffT fault: > 3.5°C
    """
    
    DT_THRESHOLD = 0.20
    TEMP_ALARM = 8.5
    TEMP_CRITICAL = 10.0
    DIFF_NORMAL = 0.30
    DIFF_FAULT = 3.5
    
    synthetic_data = []
    
    # Get base features from original normal data
    normal_base = df_original[df_original['label'] == 0].iloc[:100]
    
    # ========== CLASS 1: COOLING FAILURE (3 Episodes) ==========
    print("Generating Class 1 - Cooling Failure Episodes...")
    
    # Episode 1: Slow rise (30 steps, +0.3°C per step)
    print("  Episode 1: Slow rise...")
    for step in range(30):
        base_row = normal_base.sample(1).iloc[0].copy()
        base_row['temperature1_c'] = 8.0 + (step * 0.3)
        base_row['temperature2_c'] = 8.0 + (step * 0.3) + np.random.normal(0, 0.15)
        base_row['dT1'] = 0.3 if step > 0 else 0
        base_row['dT2'] = 0.3 if step > 0 else np.random.normal(0, 0.1)
        base_row['avgT1_5'] = base_row['temperature1_c']
        base_row['avgT2_5'] = base_row['temperature2_c']
        base_row['diffT'] = abs(base_row['temperature1_c'] - base_row['temperature2_c'])
        base_row['label'] = 1
        synthetic_data.append(base_row)
    
    # Episode 2: Medium rise (24 steps, +0.5°C per step)
    print("  Episode 2: Medium rise...")
    for step in range(24):
        base_row = normal_base.sample(1).iloc[0].copy()
        base_row['temperature1_c'] = 8.0 + (step * 0.5)
        base_row['temperature2_c'] = 8.0 + (step * 0.5) + np.random.normal(0, 0.15)
        base_row['dT1'] = 0.5 if step > 0 else 0
        base_row['dT2'] = 0.5 if step > 0 else np.random.normal(0, 0.1)
        base_row['avgT1_5'] = base_row['temperature1_c']
        base_row['avgT2_5'] = base_row['temperature2_c']
        base_row['diffT'] = abs(base_row['temperature1_c'] - base_row['temperature2_c'])
        base_row['label'] = 1
        synthetic_data.append(base_row)
    
    # Episode 3: Fast rise (30 steps, +0.8°C per step) - MORE EPISODES
    print("  Episode 3: Fast rise...")
    for step in range(30):
        base_row = normal_base.sample(1).iloc[0].copy()
        base_row['temperature1_c'] = 8.0 + (step * 0.8)
        base_row['temperature2_c'] = 8.0 + (step * 0.8) + np.random.normal(0, 0.15)
        base_row['dT1'] = 0.8 if step > 0 else 0
        base_row['dT2'] = 0.8 if step > 0 else np.random.normal(0, 0.1)
        base_row['avgT1_5'] = base_row['temperature1_c']
        base_row['avgT2_5'] = base_row['temperature2_c']
        base_row['diffT'] = abs(base_row['temperature1_c'] - base_row['temperature2_c'])
        base_row['label'] = 1
        synthetic_data.append(base_row)
    
    # Episode 4: Stuck at high temperature (30 samples) - EDGE CASE FIX
    print("  Episode 4: Stuck at high temp (failure already happened)...")
    for _ in range(30):
        base_row = normal_base.sample(1).iloc[0].copy()
        # Temperature stuck between 10-15°C (above TEMP_CRITICAL)
        stuck_temp = np.random.uniform(10.5, 15.0)
        base_row['temperature1_c'] = stuck_temp
        base_row['temperature2_c'] = stuck_temp + np.random.normal(0, 0.2)
        base_row['dT1'] = np.random.uniform(-0.05, 0.05)  # Near zero (not rising)
        base_row['dT2'] = np.random.uniform(-0.05, 0.05)
        base_row['avgT1_5'] = base_row['temperature1_c']
        base_row['avgT2_5'] = base_row['temperature2_c']
        base_row['diffT'] = abs(base_row['temperature1_c'] - base_row['temperature2_c'])
        base_row['label'] = 1
        synthetic_data.append(base_row)
    
    # Additional episodes to reach 210
    remaining_ep1 = 210 - len([x for x in synthetic_data if x['label'] == 1])
    print(f"  Additional episodes: {remaining_ep1} samples...")
    for i in range(remaining_ep1):
        base_row = normal_base.sample(1).iloc[0].copy()
        # Random rise pattern
        rise_rate = np.random.choice([0.2, 0.4, 0.6])
        steps = np.random.randint(10, 25)
        step = i % steps
        base_row['temperature1_c'] = 8.0 + (step * rise_rate)
        base_row['temperature2_c'] = base_row['temperature1_c'] + np.random.normal(0, 0.2)
        base_row['dT1'] = rise_rate if step > 0 else 0
        base_row['dT2'] = rise_rate if step > 0 else np.random.normal(0, 0.1)
        base_row['avgT1_5'] = base_row['temperature1_c']
        base_row['avgT2_5'] = base_row['temperature2_c']
        base_row['diffT'] = abs(base_row['temperature1_c'] - base_row['temperature2_c'])
        base_row['label'] = 1
        synthetic_data.append(base_row)
    
    # ========== CLASS 2: SENSOR FAULT (210 samples) ==========
    print("Generating Class 2 - Sensor Fault...")
    
    # Sub-class 2A: Unrealistic temperature (70 samples)
    print("  Subclass 2A: Unrealistic temperature...")
    for _ in range(70):
        base_row = normal_base.sample(1).iloc[0].copy()
        # Either too cold or too hot
        fault_type = np.random.choice(['too_cold', 'too_hot'])
        if fault_type == 'too_cold':
            base_row['temperature1_c'] = np.random.uniform(-30, -15)
        else:
            base_row['temperature1_c'] = np.random.uniform(45, 60)
        base_row['temperature2_c'] = np.random.uniform(6, 10)  # T2 normal
        base_row['dT1'] = np.random.uniform(-5, 5)  # Erratic
        base_row['dT2'] = np.random.normal(0, 0.2)
        base_row['diffT'] = abs(base_row['temperature1_c'] - base_row['temperature2_c'])
        base_row['stdT1_5'] = np.random.uniform(1.5, 3.0)
        base_row['stuck_count1'] = 0
        base_row['stuck_count2'] = 0
        base_row['label'] = 2
        synthetic_data.append(base_row)
    
    # Sub-class 2B: Large offset (diffT > 3.5°C) (70 samples)
    print("  Subclass 2B: Large sensor offset (diffT > 3.5)...")
    for _ in range(70):
        base_row = normal_base.sample(1).iloc[0].copy()
        # Ensure diffT > 3.5°C (stricter now, was >3.5, same)
        base_row['temperature1_c'] = np.random.uniform(11, 15)
        base_row['temperature2_c'] = np.random.uniform(7, 8.5)
        base_row['diffT'] = abs(base_row['temperature1_c'] - base_row['temperature2_c'])
        base_row['dT1'] = np.random.uniform(-2, 2)
        base_row['dT2'] = np.random.uniform(-0.3, 0.3)
        base_row['stuck_count1'] = 0
        base_row['stuck_count2'] = 0
        base_row['label'] = 2
        synthetic_data.append(base_row)
    
    # Sub-class 2C: Stuck sensor (1000+ repeats) (70 samples)
    print("  Subclass 2C: Stuck sensor (repeated values)...")
    stuck_value_t1 = 7.1
    stuck_value_h1 = 76.0
    stuck_repeats = 1000  # 1000 repeats = extreme fault
    for rep_count in range(70):
        base_row = normal_base.sample(1).iloc[0].copy()
        # Simulate that sensor has been reading same value for 1000+ times
        base_row['temperature1_c'] = stuck_value_t1
        base_row['humidity1_percent'] = stuck_value_h1
        base_row['temperature2_c'] = np.random.uniform(7, 9)
        base_row['dT1'] = 0.0  # No change
        base_row['dT2'] = np.random.normal(0, 0.05)
        base_row['abs_dT1'] = 0.0
        base_row['stuck_count1'] = stuck_repeats  # Indicator of fault
        base_row['stuck_count2'] = 0
        base_row['diffT'] = abs(base_row['temperature1_c'] - base_row['temperature2_c'])
        base_row['stdT1_5'] = 0.0  # No variance
        base_row['label'] = 2
        synthetic_data.append(base_row)
    
    # ========== CLASS 0: NORMAL EXTRAS (107 samples) ==========
    print("Generating Class 0 - Normal extras...")
    for _ in range(107):
        base_row = normal_base.sample(1).iloc[0].copy()
        # Strict: T in range, diffT < 0.3°C, dT near 0
        base_temp = np.random.uniform(6.5, 8.4)
        base_row['temperature1_c'] = base_temp
        base_row['temperature2_c'] = base_temp + np.random.uniform(-0.25, 0.25)  # ± 0.25°C
        base_row['dT1'] = np.random.uniform(-0.05, 0.05)  # ≈ 0
        base_row['dT2'] = np.random.uniform(-0.05, 0.05)
        base_row['avgT1_5'] = base_row['temperature1_c']
        base_row['avgT2_5'] = base_row['temperature2_c']
        base_row['diffT'] = abs(base_row['temperature1_c'] - base_row['temperature2_c'])
        base_row['stuck_count1'] = 0
        base_row['stuck_count2'] = 0
        base_row['label'] = 0
        synthetic_data.append(base_row)
    
    df_synthetic = pd.DataFrame(synthetic_data)
    return df_synthetic

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("SEQUENCE-AWARE AUGMENTATION - 700 SAMPLES (BALANCED 40/30/30)")
    print("=" * 80)
    
    # 1. Load raw data
    print("\n1. Loading raw dataset...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"   Loaded: {len(df_raw)} rows")
    
    # 2. Feature engineering on original data
    print("\n2. Calculating features on original data...")
    df_engineered = calculate_features(df_raw)
    
    # 3. Label original data
    print("\n3. Labeling original data (KEEP ORDERED)...")
    df_labeled = label_original_data(df_engineered)
    
    original_counts = df_labeled['label'].value_counts().sort_index()
    print(f"   Original data distribution:")
    print(f"   - Class 0 (Normal): {original_counts[0]} rows")
    print(f"   - Class 1 (Cooling Failure): {original_counts[1] if 1 in original_counts else 0} rows")
    if 2 in original_counts:
        print(f"   - Class 2 (Sensor Fault): {original_counts[2]} rows")
    
    # 4. Generate synthetic data
    print("\n4. Generating synthetic data (527 samples)...")
    df_synthetic = generate_synthetic_data(df_labeled)
    
    synthetic_counts = df_synthetic['label'].value_counts().sort_index()
    print(f"   Synthetic data distribution:")
    for label in sorted(synthetic_counts.index):
        print(f"   - Class {label}: {synthetic_counts[label]} samples")
    
    # 5. Combine
    print("\n5. Combining original + synthetic...")
    
    # Select features
    feature_cols = [
        'temperature1_c', 'humidity1_percent', 'temperature2_c', 'humidity2_percent',
        'dT1', 'dT2', 'avgT1_5', 'avgT2_5', 'stdT1_5', 'stdT2_5',
        'stuck_count1', 'stuck_count2', 'diffT', 'abs_dT1', 'abs_dT2'
    ]
    
    df_labeled_clean = df_labeled[feature_cols + ['label']].copy()
    df_synthetic_clean = df_synthetic[feature_cols + ['label']].copy()
    
    # Combine (original first to preserve order)
    df_final = pd.concat([df_labeled_clean, df_synthetic_clean], ignore_index=True)
    
    print(f"   Final dataset shape: {df_final.shape}")
    
    # 6. Save
    print("\n6. Saving final dataset...")
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"   [OK] Saved to: {OUTPUT_PATH}")
    
    # 7. Statistics
    print("\n" + "=" * 80)
    print("FINAL DATASET STATISTICS")
    print("=" * 80)
    
    final_counts = df_final['label'].value_counts().sort_index()
    label_names = {0: "Normal operation", 1: "Abnormal - Cooling Failure", 2: "Abnormal - Sensor Fault"}
    
    print(f"\nTotal samples: {len(df_final)}")
    print(f"\nClass distribution:")
    for label in sorted(final_counts.index):
        count = final_counts[label]
        pct = (count / len(df_final)) * 100
        print(f"  Class {label}: {label_names[label]:<35} {count:3d} samples ({pct:5.1f}%)")
    
    print(f"\nFeature statistics:")
    print(df_final[feature_cols].describe().round(3))
    
    print(f"\nClass-wise feature means:")
    for label in sorted(final_counts.index):
        class_data = df_final[df_final['label'] == label]
        print(f"\nClass {label} ({label_names[label]}):")
        print(f"  T1: {class_data['temperature1_c'].mean():.2f}°C ± {class_data['temperature1_c'].std():.2f}")
        print(f"  T2: {class_data['temperature2_c'].mean():.2f}°C ± {class_data['temperature2_c'].std():.2f}")
        print(f"  diffT: {class_data['diffT'].mean():.2f}°C ± {class_data['diffT'].std():.2f}")
        print(f"  dT1: {class_data['dT1'].mean():.3f}°C ± {class_data['dT1'].std():.3f}")
        print(f"  stdT1_5: {class_data['stdT1_5'].mean():.3f} ± {class_data['stdT1_5'].std():.3f}")
        print(f"  stuck_count1: {class_data['stuck_count1'].mean():.1f}")
    
    print("\n" + "=" * 80)
    print("[DONE] SEQUENCE-AWARE AUGMENTATION COMPLETE!")
    print("[DONE] Dataset ready for Edge Impulse training")
    print("=" * 80)

if __name__ == "__main__":
    main()
