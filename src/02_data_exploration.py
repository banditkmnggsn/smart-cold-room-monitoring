"""
Data Exploration & Visualization
Untuk memahami distribusi dataset sebelum labeling
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

AUGMENTED_DATA_PATH = os.path.join(DATA_DIR, "dataset_engineered_augmented.csv")
OUTPUT_PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")

# Create plots directory
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

def explore_data():
    """Load dan explore dataset"""
    print("=" * 70)
    print("DATA EXPLORATION")
    print("=" * 70)
    
    df = pd.read_csv(AUGMENTED_DATA_PATH)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nLabel distribution:")
    label_map = {0: 'NORMAL', 1: 'WARM_EXCURSION', 2: 'SENSOR_FAULT'}
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        print(f"  {label_map[int(label)]}: {count} samples")
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df

def visualize_features(df):
    """Visualize feature distributions per label"""
    print("\n" + "=" * 70)
    print("VISUALIZING FEATURES")
    print("=" * 70)
    
    label_map = {0: 'NORMAL', 1: 'WARM_EXCURSION', 2: 'SENSOR_FAULT'}
    features_to_plot = [
        'temperature1_c', 'temperature2_c', 'humidity1_percent',
        'dT1', 'dT2', 'diffT', 'avgT1_5', 'avgT2_5',
        'stdT1_5', 'stdT2_5', 'stuck_count1', 'stuck_count2'
    ]
    
    # 1. Distribution per label (box plot)
    print("Creating box plots...")
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle('Feature Distribution by Class', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        for label in sorted(df['label'].unique()):
            data = df[df['label'] == label][feature]
            parts = ax.boxplot([data], positions=[label], widths=0.5, patch_artist=True,
                              labels=[label_map[int(label)]])
            for patch in parts['boxes']:
                patch.set_facecolor(['#3498db', '#e74c3c', '#f39c12'][int(label)])
        
        ax.set_title(feature, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, '01_features_boxplot.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()
    
    # 2. Feature correlation heatmap
    print("Creating correlation heatmap...")
    numeric_cols = [c for c in df.columns if c not in ['index', 'timestamp_ms', 'status', 'label']]
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                ax=ax, square=True, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, '02_correlation_heatmap.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()
    
    # 3. Key features scatter plot
    print("Creating scatter plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Key Features for Classification', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    scatter_pairs = [
        ('diffT', 'temperature1_c'),
        ('diffT', 'dT1'),
        ('stuck_count1', 'stdT1_5'),
        ('avgT1_5', 'avgT2_5'),
        ('dT1', 'dT2'),
        ('stdT1_5', 'stdT2_5')
    ]
    
    colors = {0: '#3498db', 1: '#e74c3c', 2: '#f39c12'}
    
    for idx, (feat1, feat2) in enumerate(scatter_pairs):
        ax = axes[idx]
        for label in sorted(df['label'].unique()):
            data = df[df['label'] == label]
            ax.scatter(data[feat1], data[feat2], c=colors[int(label)], 
                      label=label_map[int(label)], alpha=0.6, s=50)
        ax.set_xlabel(feat1, fontweight='bold')
        ax.set_ylabel(feat2, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, '03_scatter_plots.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()
    
    # 4. Distribution histograms
    print("Creating histograms...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Feature Distributions by Class', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, feature in enumerate(features_to_plot[:9]):
        ax = axes[idx]
        for label in sorted(df['label'].unique()):
            data = df[df['label'] == label][feature]
            ax.hist(data, bins=15, alpha=0.5, label=label_map[int(label)], 
                   color=colors[int(label)])
        ax.set_title(feature, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, '04_histograms.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()

def generate_summary(df):
    """Generate summary statistics"""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS PER CLASS")
    print("=" * 70)
    
    label_map = {0: 'NORMAL', 1: 'WARM_EXCURSION', 2: 'SENSOR_FAULT'}
    key_features = ['temperature1_c', 'temperature2_c', 'dT1', 'dT2', 'diffT', 'stdT1_5', 'stdT2_5']
    
    for label in sorted(df['label'].unique()):
        print(f"\n{label_map[int(label)]} (label={label}):")
        class_data = df[df['label'] == label][key_features]
        print(class_data.describe().round(3))

def create_feature_importance_plot(df):
    """Create feature importance based on class separation"""
    print("\nCalculating feature importance...")
    
    label_map = {0: 'NORMAL', 1: 'WARM_EXCURSION', 2: 'SENSOR_FAULT'}
    numeric_cols = [c for c in df.columns if c not in ['index', 'timestamp_ms', 'status', 'label']]
    
    # Calculate variance ratio (feature importance indicator)
    importance_scores = []
    
    for feature in numeric_cols:
        # Calculate between-class variance / within-class variance
        overall_mean = df[feature].mean()
        
        between_var = 0
        within_var = 0
        
        for label in sorted(df['label'].unique()):
            class_data = df[df['label'] == label][feature]
            class_mean = class_data.mean()
            class_count = len(class_data)
            
            # Between-class variance
            between_var += class_count * (class_mean - overall_mean) ** 2
            
            # Within-class variance
            within_var += ((class_data - class_mean) ** 2).sum()
        
        # F-ratio as importance score
        within_var = within_var / max(1, len(df) - 3)
        between_var = between_var / 2
        f_ratio = between_var / max(within_var, 1e-6)
        
        importance_scores.append((feature, f_ratio))
    
    # Sort by importance
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    features_sorted = [f[0] for f in importance_scores]
    scores_sorted = [f[1] for f in importance_scores]
    
    bars = ax.barh(features_sorted, scores_sorted, color='#3498db')
    ax.set_xlabel('F-Ratio (Feature Importance)', fontweight='bold')
    ax.set_title('Feature Importance for Classification', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Color top 5 features
    for i in range(min(5, len(bars))):
        bars[i].set_color('#e74c3c')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, '05_feature_importance.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()
    
    print("\nTop 5 Most Important Features:")
    for idx, (feature, score) in enumerate(importance_scores[:5], 1):
        print(f"  {idx}. {feature}: {score:.2f}")

def main():
    # Explore data
    df = explore_data()
    
    # Generate visualizations
    visualize_features(df)
    
    # Summary statistics
    generate_summary(df)
    
    # Feature importance
    create_feature_importance_plot(df)
    
    print("\n" + "=" * 70)
    print("✓ Exploration complete! Check plots/ folder for visualizations")
    print("=" * 70)

if __name__ == "__main__":
    main()
