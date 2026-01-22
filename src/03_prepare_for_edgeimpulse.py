"""
Prepare dataset untuk Edge Impulse / labeling
Export dalam format yang siap pakai
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

AUGMENTED_DATA_PATH = os.path.join(DATA_DIR, "dataset_engineered_augmented.csv")
EDGE_IMPULSE_PATH = os.path.join(DATA_DIR, "dataset_for_edgeimpulse.csv")
LABELING_GUIDE_PATH = os.path.join(DATA_DIR, "LABELING_GUIDE.txt")

def prepare_for_edge_impulse():
    """
    Prepare dataset for Edge Impulse export
    Format: X1, X2, X3, ..., X15, y
    """
    
    print("=" * 70)
    print("PREPARING DATASET FOR EDGE IMPULSE")
    print("=" * 70)
    
    df = pd.read_csv(AUGMENTED_DATA_PATH)
    
    # Select features (skip index, timestamp, status)
    feature_cols = [
        'temperature1_c', 'humidity1_percent', 'temperature2_c', 'humidity2_percent',
        'dT1', 'dT2', 'avgT1_5', 'avgT2_5', 'stdT1_5', 'stdT2_5',
        'stuck_count1', 'stuck_count2', 'diffT', 'abs_dT1', 'abs_dT2'
    ]
    
    # Create clean dataset
    df_export = df[feature_cols + ['label']].copy()
    
    # Rename label to target
    df_export = df_export.rename(columns={'label': 'y'})
    
    # Save
    df_export.to_csv(EDGE_IMPULSE_PATH, index=False)
    print(f"\n✓ Exported to: {EDGE_IMPULSE_PATH}")
    print(f"  Shape: {df_export.shape}")
    print(f"\nFeatures ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nLabel distribution:")
    label_map = {0: 'NORMAL', 1: 'WARM_EXCURSION', 2: 'SENSOR_FAULT'}
    for label in sorted(df_export['y'].unique()):
        count = (df_export['y'] == label).sum()
        print(f"  {label}: {label_map[label]} - {count} samples")
    
    # Create labeling guide
    create_labeling_guide(label_map)
    
    return df_export

def create_labeling_guide(label_map):
    """Create a guide for understanding the labels"""
    
    guide = """
================================================================================
                    LABELING GUIDE - TinyML Cold Room Monitoring
================================================================================

TARGET VARIABLE: y (3-class classification)
- 0: NORMAL
- 1: WARM_EXCURSION  
- 2: SENSOR_FAULT

================================================================================
CLASS DESCRIPTIONS
================================================================================

Class 0: NORMAL
---------
Kondisi ruangan dingin SEHAT. Semua sensor berfungsi normal, suhu stabil.

Karakteristik:
• Suhu stabil: dT1, dT2 < 0.1°C (perubahan minimal antar pembacaan)
• Sensor konsisten: diffT < 0.5°C (kedua sensor membaca hal yang sama)
• Tidak ada sensor stuck: stuck_count1, stuck_count2 ≈ 0
• Humidity normal: 70-95%
• Standard deviasi kecil: stdT1_5, stdT2_5 < 0.3

Contoh kasus:
- Suhu T1 = 7.1°C, T2 = 7.2°C → NORMAL
- Suhu stabil selama beberapa jam
- Sensor bergerak dengan mulus


Class 1: WARM_EXCURSION
----------------------
Ruangan TERLALU PANAS! Suhu naik lebih tinggi dari normal. Ini adalah alarm!

Karakteristik:
• Suhu lebih tinggi dari normal: temperature1_c, temperature2_c > 10-11°C
• Trend naik: dT1, dT2 > 0.1°C (positif terus-menerus)
• Sensor masih konsisten: diffT < 0.5°C (kedua sensor menunjukkan trend yang sama)
• Tidak ada perubahan drastis (bukan spike)

Contoh kasus:
- Suhu naik dari 8°C → 10.5°C secara perlahan
- Perubahan rate: +0.2°C setiap pembacaan
- Alert: "Cold room too warm! Check equipment!"
- Alarm trigger untuk teknisi


Class 2: SENSOR_FAULT
--------------------
ADA MASALAH DENGAN SENSOR. Bisa DHT11 stuck, spike unrealistic, atau offset besar.

Karakteristik:
• Perbedaan besar antar sensor: diffT > 2.0°C (tidak masuk akal)
  - Contoh: T1 = 12°C tapi T2 = 7°C (beda 5°C = salah satu rusak)
  
ATAU
  
• Sensor stuck (tidak berubah): stuck_count1 atau stuck_count2 > 3
  - Nilai sensor = 7.1°C selama 5+ pembacaan berturut-turut
  
ATAU
  
• Spike unrealistic: abs_dT > 2°C (lompat tiba-tiba)
  - Contoh: T1 dari 7.1°C → 12.5°C dalam 1 pembacaan
  - Standard deviasi tinggi: stdT1_5 > 1.5
  
ATAU

• High variability: stdT1_5 atau stdT2_5 > 1.0
  - Data sangat berisik / tidak konsisten

Contoh kasus:
- T1 = 12°C, T2 = 7°C (diffT = 5°C) → sensor T1 mungkin rusak
- T1 stuck di 7.1°C selama 6 pembacaan → DHT11 stuck
- T1 naik 4°C dalam 1 frame, stdT1_5 = 2.5 → spike sensor
- Alert: "Sensor fault detected! Replace DHT11"


================================================================================
FEATURE DEFINITIONS
================================================================================

Raw Features:
• temperature1_c       : Suhu sensor 1 (°C)
• temperature2_c       : Suhu sensor 2 (°C)
• humidity1_percent    : Kelembaban sensor 1 (%)
• humidity2_percent    : Kelembaban sensor 2 (%)

Engineered Features:
• dT1                  : Perubahan suhu sensor 1 (T1_now - T1_prev)
• dT2                  : Perubahan suhu sensor 2 (T2_now - T2_prev)
• avgT1_5              : Rata-rata suhu sensor 1 (5 data terakhir)
• avgT2_5              : Rata-rata suhu sensor 2 (5 data terakhir)
• stdT1_5              : Standar deviasi suhu sensor 1 (5 data terakhir)
• stdT2_5              : Standar deviasi suhu sensor 2 (5 data terakhir)
• stuck_count1         : Jumlah pembacaan T1 stabil (<0.1°C change)
• stuck_count2         : Jumlah pembacaan T2 stabil (<0.1°C change)
• diffT                : abs(T1 - T2) - Perbedaan antar sensor
• abs_dT1              : abs(dT1) - Magnitude perubahan T1
• abs_dT2              : abs(dT2) - Magnitude perubahan T2


================================================================================
IMPORTANT NOTES
================================================================================

1. PRIORITAS FITUR untuk klasifikasi:
   ✓ diffT (paling penting!) - deteksi offset sensor
   ✓ dT1, dT2 - deteksi trend vs spike
   ✓ stdT1_5, stdT2_5 - deteksi noise/unstable
   ✓ stuck_count - deteksi sensor stuck
   ✓ temperature1_c, temperature2_c - deteksi warm excursion

2. EDGE CASES:
   • Jika diffT = 5°C tapi dT1 = 0, dT2 = 0 → Likely SENSOR_FAULT
   • Jika dT1 = +0.3, dT2 = +0.3 secara terus → WARM_EXCURSION
   • Jika dT1 = -3°C dalam 1 frame → SENSOR_FAULT (spike)

3. UNTUK MODEL TRAINING:
   • Dataset sudah balanced: 100 samples per class
   • Features sudah engineered: siap untuk neural network
   • Siap di-export ke Edge Impulse untuk training
   • Atau bisa langsung train di Python dengan TensorFlow

================================================================================
"""
    
    with open(LABELING_GUIDE_PATH, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"\n✓ Created labeling guide: {LABELING_GUIDE_PATH}")

def show_sample_data():
    """Show sample data per class"""
    
    print("\n" + "=" * 70)
    print("SAMPLE DATA PER CLASS")
    print("=" * 70)
    
    df = pd.read_csv(EDGE_IMPULSE_PATH)
    
    label_map = {0: 'NORMAL', 1: 'WARM_EXCURSION', 2: 'SENSOR_FAULT'}
    
    for label in sorted(df['y'].unique()):
        print(f"\n{label_map[label]} (y={label}):")
        sample = df[df['y'] == label].head(3)
        print(sample.to_string())

def main():
    df = prepare_for_edge_impulse()
    show_sample_data()
    
    print("\n" + "=" * 70)
    print("✓ READY FOR EDGE IMPULSE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Use dataset_for_edgeimpulse.csv for Edge Impulse training")
    print("2. Or use dataset_engineered_augmented.csv for local training")
    print("3. See LABELING_GUIDE.txt for class definitions")

if __name__ == "__main__":
    main()
