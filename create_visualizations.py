"""
Create comprehensive visualizations for the statistical and fuzzy papers.

This script generates:
1. Boxplots for all features comparing human vs LLM
2. ROC curves for all classifiers
3. PR curves for all classifiers
4. Feature correlation heatmap
5. Fuzzy membership functions visualization
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
print("Loading data...")
features_df = pd.read_csv('features_100k.csv')
test_results = pd.read_csv('statistical_tests_results.csv')

with open('roc_results.pkl', 'rb') as f:
    roc_results = pickle.load(f)
with open('pr_results.pkl', 'rb') as f:
    pr_results = pickle.load(f)
with open('fuzzy_roc_results.pkl', 'rb') as f:
    fuzzy_roc = pickle.load(f)
with open('fuzzy_pr_results.pkl', 'rb') as f:
    fuzzy_pr = pickle.load(f)

# ============================================================================
# 1. Boxplots for all features
# ============================================================================
print("Creating boxplots...")

# Select features (exclude fk_grade which is all zeros)
feature_cols = [c for c in features_df.columns if c not in ['label', 'fk_grade']]

# Create figure with subplots
n_features = len(feature_cols)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
axes = axes.flatten()

for idx, feature in enumerate(feature_cols):
    ax = axes[idx]

    # Prepare data
    human_data = features_df[features_df['label'] == 'human'][feature]
    llm_data = features_df[features_df['label'] == 'llm'][feature]

    # Create boxplot
    bp = ax.boxplot([human_data, llm_data],
                     patch_artist=True,
                     showfliers=False)
    ax.set_xticklabels(['Humano', 'LLM'])

    # Color boxes
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Get test results for this feature
    test_row = test_results[test_results['feature'] == feature].iloc[0]
    p_val = test_row['p_value']
    delta = test_row['delta']

    # Add title with statistics
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax.set_title(f'{feature}\n(δ={delta:.3f}, {sig})', fontsize=9)
    ax.set_ylabel('Valor')

# Remove empty subplots
for idx in range(len(feature_cols), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('figure_boxplots.png', dpi=300, bbox_inches='tight')
print("Saved: figure_boxplots.png")
plt.close()

# ============================================================================
# 2. ROC Curves
# ============================================================================
print("Creating ROC curves...")

fig, ax = plt.subplots(figsize=(8, 7))

# Plot each classifier
colors = {'LDA': '#2ecc71', 'Logistic': '#3498db', 'Fuzzy': '#e74c3c'}
linestyles = {'LDA': '-', 'Logistic': '--', 'Fuzzy': '-.'}

# Traditional classifiers
for model_name in ['LDA', 'Logistic']:
    all_fpr = []
    all_tpr = []
    aucs = []

    for fold in roc_results[model_name]:
        all_fpr.append(fold['fpr'])
        all_tpr.append(fold['tpr'])
        aucs.append(fold['auc'])

    # Plot mean curve
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Interpolate to common FPR grid
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    for fpr, tpr in zip(all_fpr, all_tpr):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tprs.append(interp_tpr)

    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr = np.std(interp_tprs, axis=0)

    ax.plot(mean_fpr, mean_tpr,
            color=colors[model_name],
            linestyle=linestyles[model_name],
            linewidth=2,
            label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    # Add confidence band
    ax.fill_between(mean_fpr,
                    mean_tpr - std_tpr,
                    mean_tpr + std_tpr,
                    color=colors[model_name],
                    alpha=0.2)

# Fuzzy classifier
fuzzy_fpr_list = []
fuzzy_tpr_list = []
fuzzy_aucs = []

for fold in fuzzy_roc['Fuzzy']:
    fuzzy_fpr_list.append(fold['fpr'])
    fuzzy_tpr_list.append(fold['tpr'])
    fuzzy_aucs.append(fold['auc'])

mean_auc_fuzzy = np.mean(fuzzy_aucs)
std_auc_fuzzy = np.std(fuzzy_aucs)

# Interpolate
mean_fpr = np.linspace(0, 1, 100)
interp_tprs_fuzzy = []
for fpr, tpr in zip(fuzzy_fpr_list, fuzzy_tpr_list):
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tprs_fuzzy.append(interp_tpr)

mean_tpr_fuzzy = np.mean(interp_tprs_fuzzy, axis=0)
std_tpr_fuzzy = np.std(interp_tprs_fuzzy, axis=0)

ax.plot(mean_fpr, mean_tpr_fuzzy,
        color=colors['Fuzzy'],
        linestyle=linestyles['Fuzzy'],
        linewidth=2,
        label=f'Fuzzy (AUC = {mean_auc_fuzzy:.3f} ± {std_auc_fuzzy:.3f})')

ax.fill_between(mean_fpr,
                mean_tpr_fuzzy - std_tpr_fuzzy,
                mean_tpr_fuzzy + std_tpr_fuzzy,
                color=colors['Fuzzy'],
                alpha=0.2)

# Diagonal line
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório (AUC = 0.500)')

ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR / Recall)', fontsize=12)
ax.set_title('Curvas ROC: Classificação de Textos Humanos vs LLM', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('figure_roc_curves.png', dpi=300, bbox_inches='tight')
print("Saved: figure_roc_curves.png")
plt.close()

# ============================================================================
# 3. Precision-Recall Curves
# ============================================================================
print("Creating PR curves...")

fig, ax = plt.subplots(figsize=(8, 7))

# Traditional classifiers
for model_name in ['LDA', 'Logistic']:
    all_precision = []
    all_recall = []
    aps = []

    for fold in pr_results[model_name]:
        all_precision.append(fold['precision'])
        all_recall.append(fold['recall'])
        aps.append(fold['ap'])

    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    # Interpolate to common recall grid
    mean_recall = np.linspace(0, 1, 100)
    interp_precisions = []
    for recall, precision in zip(all_recall, all_precision):
        # Reverse for interpolation
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        interp_precisions.append(interp_precision)

    mean_precision = np.mean(interp_precisions, axis=0)
    std_precision = np.std(interp_precisions, axis=0)

    ax.plot(mean_recall, mean_precision,
            color=colors[model_name],
            linestyle=linestyles[model_name],
            linewidth=2,
            label=f'{model_name} (AP = {mean_ap:.3f} ± {std_ap:.3f})')

    ax.fill_between(mean_recall,
                    mean_precision - std_precision,
                    mean_precision + std_precision,
                    color=colors[model_name],
                    alpha=0.2)

# Fuzzy classifier
fuzzy_precision_list = []
fuzzy_recall_list = []
fuzzy_aps = []

for fold in fuzzy_pr['Fuzzy']:
    fuzzy_precision_list.append(fold['precision'])
    fuzzy_recall_list.append(fold['recall'])
    fuzzy_aps.append(fold['ap'])

mean_ap_fuzzy = np.mean(fuzzy_aps)
std_ap_fuzzy = np.std(fuzzy_aps)

mean_recall = np.linspace(0, 1, 100)
interp_precisions_fuzzy = []
for recall, precision in zip(fuzzy_recall_list, fuzzy_precision_list):
    interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
    interp_precisions_fuzzy.append(interp_precision)

mean_precision_fuzzy = np.mean(interp_precisions_fuzzy, axis=0)
std_precision_fuzzy = np.std(interp_precisions_fuzzy, axis=0)

ax.plot(mean_recall, mean_precision_fuzzy,
        color=colors['Fuzzy'],
        linestyle=linestyles['Fuzzy'],
        linewidth=2,
        label=f'Fuzzy (AP = {mean_ap_fuzzy:.3f} ± {std_ap_fuzzy:.3f})')

ax.fill_between(mean_recall,
                mean_precision_fuzzy - std_precision_fuzzy,
                mean_precision_fuzzy + std_precision_fuzzy,
                color=colors['Fuzzy'],
                alpha=0.2)

# Baseline
ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1, label='Baseline (AP = 0.500)')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precisão', fontsize=12)
ax.set_title('Curvas Precision-Recall: Classificação de Textos Humanos vs LLM', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0.48, 1.02])

plt.tight_layout()
plt.savefig('figure_pr_curves.png', dpi=300, bbox_inches='tight')
print("Saved: figure_pr_curves.png")
plt.close()

# ============================================================================
# 4. Feature Correlation Heatmap
# ============================================================================
print("Creating correlation heatmap...")

# Calculate correlation matrix (exclude label and fk_grade)
feature_cols = [c for c in features_df.columns if c not in ['label', 'fk_grade']]
corr_matrix = features_df[feature_cols].corr()

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot heatmap
sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax)

ax.set_title('Matriz de Correlação entre Características', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figure_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: figure_correlation_heatmap.png")
plt.close()

# ============================================================================
# 5. Fuzzy Membership Functions (Conceptual Example)
# ============================================================================
print("Creating fuzzy membership functions...")

# Plot conceptual fuzzy membership functions based on data quantiles
selected_features = ['char_entropy', 'ttr', 'sent_std', 'hapax_prop']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feature in enumerate(selected_features):
    ax = axes[idx]

    # Calculate quantiles for this feature
    feat_data = features_df[feature]
    q0, q25, q33, q50, q66, q75, q100 = np.percentile(feat_data, [0, 25, 33, 50, 66, 75, 100])

    # Create x values
    x = np.linspace(q0, q100, 200)

    # Create triangular membership functions
    def triangular(x, a, b, c):
        y = np.zeros_like(x)
        mask1 = (x >= a) & (x < b)
        if b != a:
            y[mask1] = (x[mask1] - a) / (b - a)
        mask2 = (x >= b) & (x <= c)
        if c != b:
            y[mask2] = (c - x[mask2]) / (c - b)
        y[x == b] = 1.0
        return np.clip(y, 0, 1)

    # Define membership functions using quantiles
    low_membership = triangular(x, q0, q25, q50)
    med_membership = triangular(x, q25, q50, q75)
    high_membership = triangular(x, q50, q75, q100)

    # Plot membership functions
    ax.plot(x, low_membership, 'b-', linewidth=2.5, label='Baixo', alpha=0.8)
    ax.plot(x, med_membership, 'g-', linewidth=2.5, label='Médio', alpha=0.8)
    ax.plot(x, high_membership, 'r-', linewidth=2.5, label='Alto', alpha=0.8)

    # Add data distribution as histogram
    ax2 = ax.twinx()
    ax2.hist(features_df[features_df['label'] == 'human'][feature],
             bins=50, alpha=0.25, color='dodgerblue', label='Humano', density=True)
    ax2.hist(features_df[features_df['label'] == 'llm'][feature],
             bins=50, alpha=0.25, color='orangered', label='LLM', density=True)
    ax2.set_ylabel('Densidade', fontsize=10, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Formatting
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_ylabel('Grau de Pertinência', fontsize=10)
    ax.set_title(f'Funções de Pertinência Fuzzy: {feature}', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([x.min(), x.max()])
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figure_fuzzy_membership_functions.png', dpi=300, bbox_inches='tight')
print("Saved: figure_fuzzy_membership_functions.png")
plt.close()

print("\n" + "="*70)
print("All visualizations created successfully!")
print("="*70)
print("\nGenerated files:")
print("  - figure_boxplots.png")
print("  - figure_roc_curves.png")
print("  - figure_pr_curves.png")
print("  - figure_correlation_heatmap.png")
print("  - figure_fuzzy_membership_functions.png")
