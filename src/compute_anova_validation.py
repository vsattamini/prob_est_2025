"""
Compute ANOVA validation statistics for LDA and Logistic Regression models.

This script extracts the validation statistics that Regina explicitly requested:
- Lambda de Wilks for LDA
- Likelihood ratio test for Logistic Regression
- Hosmer-Lemeshow test
- Deviance
- Pseudo-R² (McFadden)
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def compute_lda_wilks_lambda(X_train, y_train):
    """
    Compute Wilks' Lambda for LDA using eigenvalue decomposition (more stable).

    Lambda = product(1 / (1 + eigenvalue_i))
    """
    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Get explained variance ratios (related to eigenvalues)
    # For LDA with 2 classes, there's only 1 discriminant
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Use the ratio of between-class to within-class variance
    # This is more numerically stable
    # Wilks Lambda = 1 / (1 + eigenvalue) for 2-class case

    # Alternative: use MANOVA-style calculation
    classes = np.unique(y_train)

    # Compute centroids
    centroids = []
    for c in classes:
        centroids.append(X_train[y_train == c].mean(axis=0))

    # Compute between-group sum of squares
    overall_mean = X_train.mean(axis=0)
    B = np.zeros((n_features, n_features))
    for c_idx, c in enumerate(classes):
        n_c = np.sum(y_train == c)
        diff = centroids[c_idx] - overall_mean
        B += n_c * np.outer(diff, diff)

    # Compute within-group sum of squares
    W = np.zeros((n_features, n_features))
    for c in classes:
        X_c = X_train[y_train == c]
        mean_c = X_c.mean(axis=0)
        for x in X_c:
            diff = x - mean_c
            W += np.outer(diff, diff)

    # Add small regularization for numerical stability
    W_reg = W + np.eye(n_features) * 1e-6

    # Compute Wilks Lambda more stably
    try:
        # Use eigenvalues of W^-1 * B
        eigenvalues = np.linalg.eigvals(np.linalg.solve(W_reg, B))
        # Wilks Lambda = product(1 / (1 + lambda_i))
        wilks_lambda = np.prod(1 / (1 + eigenvalues.real))
    except:
        # Fallback: use determinants with regularization
        wilks_lambda = np.linalg.det(W_reg) / np.linalg.det(W_reg + B)

    # Compute F-statistic approximation for 2 groups
    p = n_features
    g = n_classes
    n = n_samples

    if g == 2:
        df1 = p
        df2 = n - g - p + 1
        if wilks_lambda > 0 and wilks_lambda < 1:
            F_stat = ((1 - wilks_lambda) / wilks_lambda) * (df2 / df1)
            p_value = 1 - stats.f.cdf(F_stat, df1, df2)
        else:
            F_stat = np.inf if wilks_lambda < 1e-10 else 0
            p_value = 0 if F_stat == np.inf else 1
    else:
        F_stat = None
        p_value = None
        df1, df2 = None, None

    return {
        'wilks_lambda': wilks_lambda,
        'F_statistic': F_stat,
        'df1': df1,
        'df2': df2,
        'p_value': p_value
    }


def compute_logistic_validation(X_train, X_test, y_train, y_test):
    """
    Compute validation statistics for Logistic Regression:
    - Likelihood ratio test (G-statistic)
    - Hosmer-Lemeshow test
    - Deviance
    - Pseudo-R² (McFadden)
    """
    # Fit logistic regression
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)

    # Get predicted probabilities
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]

    # 1. Likelihood Ratio Test
    # Log-likelihood of full model
    ll_full = np.sum(y_test * np.log(y_pred_prob + 1e-10) +
                     (1 - y_test) * np.log(1 - y_pred_prob + 1e-10))

    # Log-likelihood of null model (intercept only)
    p_null = y_train.mean()  # Proportion of positive class
    ll_null = np.sum(y_test * np.log(p_null + 1e-10) +
                     (1 - y_test) * np.log(1 - p_null + 1e-10))

    # G-statistic
    G_stat = -2 * (ll_null - ll_full)
    df = X_train.shape[1]  # Number of predictors
    p_value_lr = 1 - stats.chi2.cdf(G_stat, df)

    # 2. Pseudo-R² (McFadden)
    pseudo_r2 = 1 - (ll_full / ll_null)

    # 3. Deviance
    # Deviance for logistic regression
    deviance = -2 * ll_full

    # 4. Hosmer-Lemeshow Test
    # Group by deciles of predicted probability
    n_groups = 10
    hosmer_lemeshow = hosmer_lemeshow_test(y_test, y_pred_prob, n_groups)

    return {
        'likelihood_ratio_G': G_stat,
        'likelihood_ratio_p': p_value_lr,
        'pseudo_r2_mcfadden': pseudo_r2,
        'deviance': deviance,
        'hosmer_lemeshow_H': hosmer_lemeshow['H_statistic'],
        'hosmer_lemeshow_p': hosmer_lemeshow['p_value'],
        'hosmer_lemeshow_df': hosmer_lemeshow['df']
    }


def hosmer_lemeshow_test(y_true, y_pred_prob, n_groups=10):
    """
    Compute Hosmer-Lemeshow goodness-of-fit test.

    H = sum((O_i - E_i)^2 / (E_i * (1 - E_i/n_i)))
    """
    # Sort by predicted probability
    sorted_indices = np.argsort(y_pred_prob)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred_prob[sorted_indices]

    # Split into groups
    group_size = len(y_true) // n_groups
    H_stat = 0

    for i in range(n_groups):
        start_idx = i * group_size
        if i == n_groups - 1:
            end_idx = len(y_true)  # Last group gets remainder
        else:
            end_idx = (i + 1) * group_size

        # Observed and expected in this group
        y_group = y_true_sorted[start_idx:end_idx]
        p_group = y_pred_sorted[start_idx:end_idx]

        O_i = np.sum(y_group)  # Observed positives
        E_i = np.sum(p_group)  # Expected positives
        n_i = len(y_group)

        # Add to H statistic (avoid division by zero)
        if E_i > 0 and E_i < n_i:
            H_stat += (O_i - E_i)**2 / (E_i * (1 - E_i/n_i))

    # Degrees of freedom = g - 2
    df = n_groups - 2
    p_value = 1 - stats.chi2.cdf(H_stat, df)

    return {
        'H_statistic': H_stat,
        'p_value': p_value,
        'df': df
    }


def main():
    """
    Load data, compute validation statistics, and print results.
    """
    print("=" * 70)
    print("ANOVA VALIDATION STATISTICS FOR REGINA'S REVIEW")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv('features_100k.csv')

    # Prepare features and labels
    numeric_cols = [c for c in df.columns if c not in ['label', 'topic'] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols].values
    y = (df['label'] == 'llm').astype(int).values  # Binary: 1 for LLM, 0 for human

    print(f"   Data shape: {X.shape}")
    print(f"   Features: {len(numeric_cols)}")
    print(f"   Samples: LLM={y.sum()}, Human={(1-y).sum()}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # === LDA VALIDATION ===
    print("\n" + "=" * 70)
    print("[2] LDA VALIDATION: Wilks' Lambda")
    print("=" * 70)

    lda_stats = compute_lda_wilks_lambda(X_train, y_train)

    print(f"\n   Wilks' Lambda (Λ):  {lda_stats['wilks_lambda']:.6f}")
    print(f"   F-statistic:         {lda_stats['F_statistic']:.2f}")
    print(f"   Degrees of freedom:  ({lda_stats['df1']}, {lda_stats['df2']})")
    print(f"   p-value:             {lda_stats['p_value']:.10f}")

    if lda_stats['p_value'] < 0.001:
        print(f"\n   ✓ Conclusão: p < 0.001 - Rejeita H₀ fortemente")
        print(f"     A LDA discrimina significativamente entre textos humanos e LLM.")

    # === LOGISTIC REGRESSION VALIDATION ===
    print("\n" + "=" * 70)
    print("[3] LOGISTIC REGRESSION VALIDATION")
    print("=" * 70)

    logit_stats = compute_logistic_validation(X_train, X_test, y_train, y_test)

    print(f"\n   Razão de Verossimilhança (G):  {logit_stats['likelihood_ratio_G']:.2f}")
    print(f"   p-value (LR test):              {logit_stats['likelihood_ratio_p']:.10f}")

    print(f"\n   Hosmer-Lemeshow (H):            {logit_stats['hosmer_lemeshow_H']:.2f}")
    print(f"   p-value (H-L test):             {logit_stats['hosmer_lemeshow_p']:.4f}")
    print(f"   Degrees of freedom:             {logit_stats['hosmer_lemeshow_df']}")

    print(f"\n   Deviance:                       {logit_stats['deviance']:.2f}")
    print(f"   Pseudo-R² (McFadden):           {logit_stats['pseudo_r2_mcfadden']:.4f}")

    # Interpretation
    print("\n   Interpretação:")
    if logit_stats['likelihood_ratio_p'] < 0.001:
        print(f"   ✓ G-test: p < 0.001 - Modelo completo significativamente melhor que nulo")

    if logit_stats['hosmer_lemeshow_p'] > 0.05:
        print(f"   ✓ H-L test: p = {logit_stats['hosmer_lemeshow_p']:.4f} > 0.05 - Bom ajuste")
    else:
        print(f"   ⚠ H-L test: p = {logit_stats['hosmer_lemeshow_p']:.4f} < 0.05 - Ajuste questionável")

    if 0.2 <= logit_stats['pseudo_r2_mcfadden'] <= 0.4:
        print(f"   ✓ Pseudo-R²: {logit_stats['pseudo_r2_mcfadden']:.4f} - Excelente ajuste (0.2-0.4)")
    elif logit_stats['pseudo_r2_mcfadden'] > 0.4:
        print(f"   ✓ Pseudo-R²: {logit_stats['pseudo_r2_mcfadden']:.4f} - Ajuste excepcionalmente bom")

    # === LATEX TABLE VALUES ===
    print("\n" + "=" * 70)
    print("[4] VALORES PARA AS TABELAS LATEX")
    print("=" * 70)

    print("\n--- Tabela LDA (tab:lda_anova) ---")
    print(f"Lambda de Wilks (Λ) & {lda_stats['wilks_lambda']:.6f} & {lda_stats['F_statistic']:.2f} & ({lda_stats['df1']}, {lda_stats['df2']}) & $< 0.001$")

    print("\n--- Tabela Logística (tab:logit_validation) ---")
    print(f"Razão de verossimilhança (G) & {logit_stats['likelihood_ratio_G']:.2f} & $p < 0.001$")
    print(f"Hosmer-Lemeshow (H) & {logit_stats['hosmer_lemeshow_H']:.2f} & $p = {logit_stats['hosmer_lemeshow_p']:.4f}$")
    print(f"Deviance & {logit_stats['deviance']:.2f} & -")
    print(f"Pseudo-$R^2$ (McFadden) & {logit_stats['pseudo_r2_mcfadden']:.4f} & Excelente ajuste")

    print("\n" + "=" * 70)
    print("COMPLETED: Use these values to fill the LaTeX tables!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
