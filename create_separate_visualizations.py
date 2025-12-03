"""
Create separate ROC and PR curve visualizations for each paper.

- paper_fuzzy: Fuzzy classifier only
- paper_stat: LDA and Logistic Regression only
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
print("Loading data...")
with open('roc_results.pkl', 'rb') as f:
    roc_results = pickle.load(f)
with open('pr_results.pkl', 'rb') as f:
    pr_results = pickle.load(f)
with open('fuzzy_roc_results.pkl', 'rb') as f:
    fuzzy_roc = pickle.load(f)
with open('fuzzy_pr_results.pkl', 'rb') as f:
    fuzzy_pr = pickle.load(f)


# ============================================================================
# Helper function for ROC curve plotting
# ============================================================================
def plot_roc_curve(ax, model_name, results, color, linestyle):
    """Plot a single ROC curve with confidence band."""
    all_fpr = []
    all_tpr = []
    aucs = []
    
    for fold in results[model_name]:
        all_fpr.append(fold['fpr'])
        all_tpr.append(fold['tpr'])
        aucs.append(fold['auc'])
    
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
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    
    ax.fill_between(mean_fpr,
                    mean_tpr - std_tpr,
                    mean_tpr + std_tpr,
                    color=color,
                    alpha=0.2)
    
    return mean_auc, std_auc


# ============================================================================
# Helper function for PR curve plotting
# ============================================================================
def plot_pr_curve(ax, model_name, results, color, linestyle):
    """Plot a single PR curve with confidence band."""
    all_precision = []
    all_recall = []
    aps = []
    
    for fold in results[model_name]:
        all_precision.append(fold['precision'])
        all_recall.append(fold['recall'])
        aps.append(fold['ap'])
    
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)
    
    # Interpolate to common recall grid
    mean_recall = np.linspace(0, 1, 100)
    interp_precisions = []
    for recall, precision in zip(all_recall, all_precision):
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        interp_precisions.append(interp_precision)
    
    mean_precision = np.mean(interp_precisions, axis=0)
    std_precision = np.std(interp_precisions, axis=0)
    
    ax.plot(mean_recall, mean_precision,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f'{model_name} (AP = {mean_ap:.3f} ± {std_ap:.3f})')
    
    ax.fill_between(mean_recall,
                    mean_precision - std_precision,
                    mean_precision + std_precision,
                    color=color,
                    alpha=0.2)
    
    return mean_ap, std_ap


# ============================================================================
# FUZZY PAPER: ROC Curve (Fuzzy only)
# ============================================================================
print("Creating ROC curve for fuzzy paper...")

fig, ax = plt.subplots(figsize=(8, 7))

# Plot Fuzzy classifier
plot_roc_curve(ax, 'Fuzzy', fuzzy_roc, '#e74c3c', '-')

# Diagonal line
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório (AUC = 0.500)')

ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR / Recall)', fontsize=12)
ax.set_title('Curva ROC: Classificador Fuzzy', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('paper_fuzzy/figure_roc_fuzzy.png', dpi=300, bbox_inches='tight')
print("Saved: paper_fuzzy/figure_roc_fuzzy.png")
plt.close()


# ============================================================================
# FUZZY PAPER: PR Curve (Fuzzy only)
# ============================================================================
print("Creating PR curve for fuzzy paper...")

fig, ax = plt.subplots(figsize=(8, 7))

# Plot Fuzzy classifier
plot_pr_curve(ax, 'Fuzzy', fuzzy_pr, '#e74c3c', '-')

# Baseline
ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1, label='Baseline (AP = 0.500)')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precisão', fontsize=12)
ax.set_title('Curva Precision-Recall: Classificador Fuzzy', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0.48, 1.02])

plt.tight_layout()
plt.savefig('paper_fuzzy/figure_pr_fuzzy.png', dpi=300, bbox_inches='tight')
print("Saved: paper_fuzzy/figure_pr_fuzzy.png")
plt.close()


# ============================================================================
# STAT PAPER: ROC Curves (LDA and Logistic only)
# ============================================================================
print("Creating ROC curves for stat paper...")

fig, ax = plt.subplots(figsize=(8, 7))

colors = {'LDA': '#2ecc71', 'Logistic': '#3498db'}
linestyles = {'LDA': '-', 'Logistic': '--'}

for model_name in ['LDA', 'Logistic']:
    plot_roc_curve(ax, model_name, roc_results, colors[model_name], linestyles[model_name])

# Diagonal line
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório (AUC = 0.500)')

ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR / Recall)', fontsize=12)
ax.set_title('Curvas ROC: LDA e Regressão Logística', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('paper_stat/figure_roc_stat.png', dpi=300, bbox_inches='tight')
print("Saved: paper_stat/figure_roc_stat.png")
plt.close()


# ============================================================================
# STAT PAPER: PR Curves (LDA and Logistic only)
# ============================================================================
print("Creating PR curves for stat paper...")

fig, ax = plt.subplots(figsize=(8, 7))

for model_name in ['LDA', 'Logistic']:
    plot_pr_curve(ax, model_name, pr_results, colors[model_name], linestyles[model_name])

# Baseline
ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1, label='Baseline (AP = 0.500)')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precisão', fontsize=12)
ax.set_title('Curvas Precision-Recall: LDA e Regressão Logística', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0.48, 1.02])

plt.tight_layout()
plt.savefig('paper_stat/figure_pr_stat.png', dpi=300, bbox_inches='tight')
print("Saved: paper_stat/figure_pr_stat.png")
plt.close()


print("\n" + "="*70)
print("All separate visualizations created successfully!")
print("="*70)
print("\nGenerated files:")
print("  - paper_fuzzy/figure_roc_fuzzy.png (Fuzzy only)")
print("  - paper_fuzzy/figure_pr_fuzzy.png (Fuzzy only)")
print("  - paper_stat/figure_roc_stat.png (LDA + Logistic)")
print("  - paper_stat/figure_pr_stat.png (LDA + Logistic)")

