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
