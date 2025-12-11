import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Locate data file
repo_root = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(repo_root, '..', 'data')

candidates = [
    os.path.join(data_dir, 'heart.csv'),
    os.path.join(data_dir, 'heart_failure_clinical_records_dataset.csv'),
]

data_file = None
for c in candidates:
    if os.path.exists(c):
        data_file = c
        break

if data_file is None:
    print(f"Error: No data file found in {data_dir}")
    exit(1)

# Load data
df = pd.read_csv(data_file)
print(f"Dataset loaded: {data_file}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Detect target column
possible_targets = ['DEATH_EVENT', 'HeartDisease', 'target', 'Outcome', 'DEATH']
target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break

if target_col is None:
    print("Error: Could not detect target column")
    exit(1)

print(f"Target column: {target_col}\n")

# Create output directory
output_dir = os.path.join(repo_root, '..', 'reports', 'figures')
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 1. CLASS DISTRIBUTION
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
class_counts = df[target_col].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['No Heart Failure', 'Heart Failure'], class_counts.values, color=colors)
axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Pie chart
class_pct = df[target_col].value_counts(normalize=True) * 100
axes[1].pie(class_pct.values, labels=['No Heart Failure', 'Heart Failure'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: class_distribution.png")
plt.close()

# ============================================================================
# 2. FEATURE DISTRIBUTIONS BY CLASS
# ============================================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != target_col]

# Select top 8 features for visualization
if len(numeric_cols) > 8:
    numeric_cols = numeric_cols[:8]

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    # Box plot
    data_to_plot = [df[df[target_col] == 0][col].dropna(), 
                     df[df[target_col] == 1][col].dropna()]
    
    bp = axes[idx].boxplot(data_to_plot, labels=['No HF', 'HF'], patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    axes[idx].set_title(col, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Feature Distributions by Heart Failure Status', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: feature_distributions.png")
plt.close()

# ============================================================================
# 3. CORRELATION HEATMAP
# ============================================================================
numeric_df = df[numeric_cols + [target_col]].copy()
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_heatmap.png")
plt.close()

# ============================================================================
# 4. DATA SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("DATASET SUMMARY STATISTICS")
print("="*80)

print("\nClass Distribution:")
print(df[target_col].value_counts())
print(f"\nClass Percentages:")
print(df[target_col].value_counts(normalize=True) * 100)

print("\n" + "-"*80)
print("Descriptive Statistics by Class:")
print("-"*80)

for class_val in sorted(df[target_col].unique()):
    class_name = 'Heart Failure' if class_val == 1 else 'No Heart Failure'
    print(f"\n{class_name} (Class {class_val}):")
    print(df[df[target_col] == class_val][numeric_cols].describe().round(2))

# ============================================================================
# 5. SAMPLE EXAMPLES
# ============================================================================
print("\n" + "="*80)
print("SAMPLE RECORDS FROM DATASET")
print("="*80)

print("\nSample 1: No Heart Failure Case")
print("-" * 80)
sample_no_hf = df[df[target_col] == 0].iloc[0]
for col, val in sample_no_hf.items():
    print(f"  {col:25s}: {val}")

print("\n\nSample 2: Heart Failure Case")
print("-" * 80)
sample_hf = df[df[target_col] == 1].iloc[0]
for col, val in sample_hf.items():
    print(f"  {col:25s}: {val}")

# ============================================================================
# 6. SAMPLE COMPARISON TABLE
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Get samples
no_hf_sample = df[df[target_col] == 0].iloc[0]
hf_sample = df[df[target_col] == 1].iloc[0]

# Create comparison data
comparison_data = []
for col in numeric_cols:
    no_hf_val = no_hf_sample[col]
    hf_val = hf_sample[col]
    comparison_data.append([col, f"{no_hf_val:.2f}", f"{hf_val:.2f}"])

# Create table
table = ax.table(cellText=comparison_data,
                colLabels=['Feature', 'No Heart Failure', 'Heart Failure'],
                cellLoc='center',
                loc='center',
                colWidths=[0.3, 0.35, 0.35])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(3):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(comparison_data) + 1):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('#ffffff')

plt.title('Sample Comparison: Heart Failure vs No Heart Failure', 
          fontsize=14, fontweight='bold', pad=20)
plt.savefig(os.path.join(output_dir, 'sample_comparison_table.png'), dpi=300, bbox_inches='tight')
print("\n✓ Saved: sample_comparison_table.png")
plt.close()

# ============================================================================
# 7. MISSING VALUES
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df)) * 100

if missing_counts.sum() == 0:
    ax.text(0.5, 0.5, 'No Missing Values Detected', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            transform=ax.transAxes)
    ax.set_title('Missing Data Report', fontsize=14, fontweight='bold')
    ax.axis('off')
else:
    missing_data = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing Count': missing_counts.values,
        'Missing %': missing_pct.values
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    
    ax.barh(missing_data['Column'], missing_data['Missing %'], color='#e74c3c')
    ax.set_xlabel('Percentage Missing (%)')
    ax.set_title('Missing Data Report', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'missing_data_report.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: missing_data_report.png")
plt.close()

# ============================================================================
# 8. DATASET OVERVIEW
# ============================================================================
print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"\nTotal Samples: {len(df)}")
print(f"Total Features: {len(df.columns)}")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"Data Types:")
for dtype, count in df.dtypes.value_counts().items():
    print(f"  {dtype}: {count}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"\nAll visualizations saved to: {output_dir}")
print("\nGenerated files:")
print("  - class_distribution.png")
print("  - feature_distributions.png")
print("  - correlation_heatmap.png")
print("  - sample_comparison_table.png")
print("  - missing_data_report.png")
print("="*80 + "\n")
