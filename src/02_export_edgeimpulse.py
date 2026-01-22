"""
Export dataset for Edge Impulse
Final format ready for upload and training
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

INPUT_PATH = os.path.join(DATA_DIR, "dataset_engineered_augmented.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "dataset_for_edgeimpulse.csv")

def export_for_edge_impulse():
    print("=" * 80)
    print("EXPORTING FOR EDGE IMPULSE")
    print("=" * 80)
    
    # Load
    df = pd.read_csv(INPUT_PATH)
    
    start_timestamp = 0
    timestamps = [start_timestamp + (i * 5) for i in range(len(df))]
    
    df_export = df.copy()
    df_export.insert(0, 'timestamp', timestamps)
    # Keep ONLY numeric label for Edge Impulse (no label_name string column)
    
    # Save
    df_export.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✓ Exported: {OUTPUT_PATH}")
    print(f"  Shape: {df_export.shape}")
    print(f"  Columns: {len(df_export.columns)}")
    print(f"  Rows: {len(df_export)}")
    
    # Class distribution
    label_name_map = {
        0: "Normal_operation",
        1: "Abnormal_cooling_failure",
        2: "Abnormal_sensor_fault",
    }
    print(f"\nClass distribution (for Edge Impulse):")
    for label_val in sorted(df_export['label'].unique()):
        count = (df_export['label'] == label_val).sum()
        pct = (count / len(df_export)) * 100
        print(f"  {label_val} ({label_name_map[label_val]:<28}): {count:3d} samples ({pct:5.1f}%)")
    
    # Show feature names
    features = [c for c in df_export.columns if c not in ['timestamp', 'label']]
    print(f"\nFeatures ({len(features)}):")
    for i, feat in enumerate(features, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\nFirst 5 rows (edge impulse format):")
    print(df_export.head().to_string())
    
    print("\nCSV Format:")
    print("  - Column 1: timestamp (milliseconds)")
    print("  - Columns 2-16: 15 features (normalized)")
    print("  - Column 17: label (0, 1, or 2)")
    
    print("\n" + "=" * 80)
    print("✓ READY FOR EDGE IMPULSE UPLOAD")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Go to https://edgeimpulse.com and sign in")
    print("2. Create new project: 'Cold-Room-Monitoring'")
    print("3. Upload: dataset_for_edgeimpulse.csv")
    print("4. When prompted:")
    print("   - Set timestamp column to: timestamp")
    print("   - Set label column to: label")
    print("   - Leave features as auto-selected")
    print("5. Create Impulse:")
    print("   - Input: 15 features")
    print("   - Output: 3 classes (0, 1, 2)")
    print("6. Choose Learning Block: Neural Network (MLP)")
    print("7. Configure network:")
    print("   - Layer 1: Dense(12, relu)")
    print("   - Layer 2: Dense(8, relu)")
    print("   - Output: Dense(3, softmax)")
    print("8. Train and export as TFLite")
    print("9. Deploy to ESP32")

if __name__ == "__main__":
    export_for_edge_impulse()
