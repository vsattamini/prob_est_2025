# GUIA 04: Classificadores e Redução de Dimensionalidade

**Objetivo:** Explicar em profundidade os métodos multivariados usados para classificar textos humanos vs LLM.

**Público-alvo:** Pessoa com mestrado em Ciência da Computação, mas sem necessariamente domínio avançado de álgebra linear ou machine learning.

**Pré-requisitos recomendados:**
- Conceito de vetor e matriz
- Produto escalar (dot product)
- Regressão linear básica
- GUIA_01, GUIA_02, GUIA_03 lidos

---

## Índice

1. [Visão Geral: Do Univariado ao Multivariado](#1-visão-geral-do-univariado-ao-multivariado)
2. [PCA - Análise de Componentes Principais](#2-pca---análise-de-componentes-principais)
3. [LDA - Análise Discriminante Linear](#3-lda---análise-discriminante-linear)
4. [Regressão Logística](#4-regressão-logística)
5. [Random Forest](#5-random-forest)
6. [Comparação dos Métodos](#6-comparação-dos-métodos)
7. [Validação Cruzada](#7-validação-cruzada)
8. [Métricas de Avaliação](#8-métricas-de-avaliação)

---

## 1. Visão Geral: Do Univariado ao Multivariado

### 1.1 Análise Univariada vs Multivariada

**Análise univariada** (GUIA_03):
- Testa **uma característica por vez** (sent_mean, ttr, entropy, etc.)
- Pergunta: "Esta característica distingue humanos de LLMs?"
- Ferramentas: Mann-Whitney U, Cliff's delta

**Análise multivariada** (este guia):
- Combina **múltiplas características simultaneamente**
- Pergunta: "Qual a melhor combinação de características para classificar?"
- Ferramentas: PCA, LDA, Regressão Logística, Random Forest

**Por quê multivariado?**

Características individuais podem ser fracas, mas juntas são poderosas:

| Texto | TTR | Sent_mean | Classe |
|-------|-----|-----------|--------|
| A | 0.60 | 20 | Humano |
| B | 0.62 | 22 | Humano |
| C | 0.70 | 24 | **LLM** |
| D | 0.72 | 26 | **LLM** |

- TTR sozinho: separação moderada
- sent_mean sozinho: separação moderada
- **TTR + sent_mean juntos:** separação perfeita!

### 1.2 Os Quatro Métodos

| Método | Tipo | O que faz | Saída |
|--------|------|-----------|-------|
| **PCA** | Redução de dimensionalidade | Encontra direções de máxima variância | Componentes principais |
| **LDA** | Redução supervisionada | Encontra direções de máxima separação entre classes | Discriminantes lineares |
| **Regressão Logística** | Classificador linear | Modela probabilidade de classe | P(LLM \| features) |
| **Random Forest** | Classificador não-linear | Ensemble de árvores de decisão | Classe + importância |

**Workflow típico:**
1. **Exploração:** PCA (visualizar dados em 2D)
2. **Análise:** LDA (encontrar eixos discriminantes)
3. **Classificação:** Regressão Logística ou Random Forest
4. **Interpretação:** Coeficientes (LR) ou importância de features (RF)

---

## 2. PCA - Análise de Componentes Principais

### 2.1 O Que É PCA?

**PCA** (Principal Component Analysis) transforma dados de alta dimensão (10 características) em baixa dimensão (2-3 componentes), preservando a maior parte da **variância**.

**Objetivo:** Encontrar as **direções** (componentes) ao longo das quais os dados variam mais.

**Analogia:** Imagine fotografar uma nuvem de pontos 3D. PCA encontra o melhor ângulo da câmera para capturar a máxima "informação" na foto 2D.

### 2.2 Conceito Visual

**Dados originais (2D):**

```
          sent_mean
            ↑
            |     ●●
            |   ●●●●
            |  ●●●●●
            | ●●●●
            |●●●
            +----------→ ttr
```

**PCA encontra PC1 (direção de máxima variância):**

```
          sent_mean
            ↑
            |     ●●
            |   ●●●●  ╱ PC1 (principal)
            |  ●●●●● ╱
            | ●●●●  ╱
            |●●●   ╱
            +----------→ ttr
```

**PC1** = combinação linear de ttr e sent_mean que captura a maior variância.

**PC2** = perpendicular a PC1, captura a segunda maior variância.

### 2.3 Matemática do PCA (Simplificado)

**Dados:** Matriz X de tamanho (n_amostras, n_features)

```
       sent_mean  ttr  entropy  ...
Texto1    20.5   0.60   3.5
Texto2    22.0   0.65   3.6
...
```

**Passo 1:** Centralizar dados (subtrair média de cada coluna)

```python
X_centered = X - X.mean(axis=0)
```

**Passo 2:** Calcular matriz de covariância

$$
\text{Cov}(X) = \frac{1}{n-1} X^T X
$$

**Passo 3:** Encontrar autovalores e autovetores de Cov(X)

- **Autovalores** (λ₁, λ₂, ...): Magnitude da variância em cada componente
- **Autovetores** (v₁, v₂, ...): Direções dos componentes

**Passo 4:** Ordenar autovetores por autovalores (maior → menor)

**Passo 5:** Projetar dados nos k primeiros componentes

$$
X_{\text{reduzido}} = X \cdot V_k
$$

Onde $V_k$ são os k primeiros autovetores.

### 2.4 Implementação em Python

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Dados exemplo (1000 amostras, 10 características)
np.random.seed(42)
n_humanos = 500
n_llms = 500

# Simular características (humanos têm valores menores em média)
features_humanos = np.random.randn(n_humanos, 10) * 1.0 + 0.0
features_llms = np.random.randn(n_llms, 10) * 1.0 + 0.5  # shift

X = np.vstack([features_humanos, features_llms])
y = np.array([0]*n_humanos + [1]*n_llms)  # 0=humano, 1=LLM

# Passo 1: Padronizar features (média=0, std=1)
# IMPORTANTE: PCA é sensível à escala!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Passo 2: Aplicar PCA
pca = PCA(n_components=2)  # Reduzir para 2 dimensões
X_pca = pca.fit_transform(X_scaled)

# Passo 3: Analisar resultados
print("Variância explicada por componente:")
print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Total: {pca.explained_variance_ratio_.sum():.2%}")

# Passo 4: Visualizar
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.5, label='Humanos', s=20)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.5, label='LLMs', s=20)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variância)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variância)', fontsize=12)
plt.title('PCA: Redução de 10D → 2D', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()
```

### 2.5 Interpretando Componentes Principais

**Loadings** = Peso de cada feature original em cada componente.

```python
# Componentes (autovetores)
componentes = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=['sent_mean', 'sent_std', 'sent_burst', 'ttr', 'herdan_c',
           'hapax_prop', 'char_entropy', 'func_word_ratio',
           'first_person_ratio', 'bigram_repeat_ratio']
)

print(componentes)
```

**Saída exemplo:**
```
                       PC1    PC2
sent_mean            0.35   0.10
sent_std            -0.30  -0.15
sent_burst          -0.32  -0.12
ttr                  0.40   0.05
herdan_c             0.38   0.08
hapax_prop           0.35   0.12
char_entropy         0.20   0.50
func_word_ratio      0.25  -0.40
first_person_ratio  -0.28   0.35
bigram_repeat_ratio -0.15  -0.30
```

**Interpretação de PC1:**
- **Positivo:** ttr, herdan_c, hapax_prop, sent_mean (características altas em LLMs)
- **Negativo:** sent_std, sent_burst, first_person_ratio (características altas em humanos)

**Conclusão:** PC1 é o "eixo LLM ← → Humano"!

### 2.6 Quantos Componentes Usar?

**Scree plot:** Gráfico de autovalores.

```python
# PCA com todos os componentes
pca_full = PCA()
pca_full.fit(X_scaled)

# Plotar
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, 11), pca_full.explained_variance_ratio_)
plt.xlabel('Componente')
plt.ylabel('Variância explicada')
plt.title('Scree Plot')

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel('Número de componentes')
plt.ylabel('Variância acumulada')
plt.title('Variância Cumulativa')
plt.axhline(0.90, color='red', linestyle='--', label='90%')
plt.legend()
plt.tight_layout()
plt.show()
```

**Regra prática:**
- Use componentes que explicam **≥ 90%** da variância total
- Ou use "elbow" (cotovelo) no scree plot

**No contexto do estudo:**
- 10 características → 2-3 componentes principais
- PC1 + PC2 frequentemente explicam 60-80% da variância

### 2.7 Limitações do PCA

- ❌ **Não supervisionado:** Ignora labels (humano/LLM)
- ❌ **Linear:** Assume relações lineares entre features
- ❌ **Sensível a escala:** Sempre padronizar antes!
- ❌ **Interpretação:** Componentes são combinações de todas as features (difícil interpretar)

**Vantagens:**
- ✅ **Visualização:** Reduz 10D → 2D para plotar
- ✅ **Remoção de correlações:** Componentes são ortogonais
- ✅ **Redução de ruído:** Descartar componentes com baixa variância

---

## 3. LDA - Análise Discriminante Linear

### 3.1 O Que É LDA?

**LDA** (Linear Discriminant Analysis) é similar ao PCA, mas **supervisionado**: encontra direções que **maximizam a separação entre classes**.

**Diferença PCA vs LDA:**

| Critério | PCA | LDA |
|----------|-----|-----|
| **Objetivo** | Maximizar variância total | Maximizar separação entre classes |
| **Usa labels?** | Não | Sim |
| **N° componentes** | min(n_features, n_amostras) | n_classes - 1 (para 2 classes: 1 LD) |

**Intuição:**

```
PCA encontra:
    ●● ○○
   ●●● ○○○  ← Máxima variância (vertical)
  ●●●● ○○○○
   ●●● ○○○
    ●● ○○

LDA encontra:
    ●● ○○
   ●●● ○○○
  ●●●● ○○○○ ← Máxima separação (horizontal)
   ●●● ○○○
    ●● ○○
```

### 3.2 Matemática do LDA (Simplificado)

**Objetivo:** Encontrar direção w que maximiza:

$$
J(w) = \frac{\text{variância entre classes}}{\text{variância dentro das classes}}
$$

**Variância entre classes** (between-class scatter):
$$
S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T
$$

**Variância dentro das classes** (within-class scatter):
$$
S_W = \sum_{i=1,2} \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T
$$

**Solução:** Autovetor correspondente ao maior autovalor de $S_W^{-1} S_B$

### 3.3 Implementação em Python

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Aplicar LDA (redução para 1 dimensão, já que 2 classes)
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

# Visualizar distribuições no eixo discriminante
plt.figure(figsize=(12, 5))

# Subplot 1: Histogramas
plt.subplot(1, 2, 1)
plt.hist(X_lda[y==0], bins=30, alpha=0.7, label='Humanos', edgecolor='black')
plt.hist(X_lda[y==1], bins=30, alpha=0.7, label='LLMs', edgecolor='black')
plt.xlabel('LD1 (Discriminante Linear 1)', fontsize=12)
plt.ylabel('Frequência')
plt.title('Distribuições no Eixo Discriminante')
plt.legend()
plt.grid(alpha=0.3)

# Subplot 2: Boxplots
plt.subplot(1, 2, 2)
data_box = [X_lda[y==0].flatten(), X_lda[y==1].flatten()]
plt.boxplot(data_box, labels=['Humanos', 'LLMs'], patch_artist=True)
plt.ylabel('LD1')
plt.title('Separação entre Classes')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Calcular separação
from scipy.stats import mannwhitneyu
u, p = mannwhitneyu(X_lda[y==0], X_lda[y==1])
print(f"Mann-Whitney U no eixo LD1: p = {p:.2e}")
```

### 3.4 Interpretando Coeficientes LDA

```python
# Coeficientes do discriminante linear
coeficientes = pd.DataFrame({
    'Característica': ['sent_mean', 'sent_std', 'sent_burst', 'ttr', 'herdan_c',
                       'hapax_prop', 'char_entropy', 'func_word_ratio',
                       'first_person_ratio', 'bigram_repeat_ratio'],
    'Coeficiente_LD1': lda.coef_[0]
})

# Ordenar por magnitude
coeficientes['Abs'] = coeficientes['Coeficiente_LD1'].abs()
coeficientes = coeficientes.sort_values('Abs', ascending=False)

print(coeficientes[['Característica', 'Coeficiente_LD1']])

# Visualizar
plt.figure(figsize=(10, 6))
plt.barh(coeficientes['Característica'], coeficientes['Coeficiente_LD1'], color='steelblue')
plt.xlabel('Coeficiente LD1', fontsize=12)
plt.title('Importância das Características no Discriminante Linear', fontsize=14)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

**Interpretação:**
- **Coeficiente positivo grande:** Característica alta → mais provável ser LLM
- **Coeficiente negativo grande:** Característica alta → mais provável ser humano

### 3.5 LDA para Classificação

LDA também pode **classificar** novos textos:

```python
# Treinar LDA como classificador
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_scaled, y)

# Predizer classes
y_pred = lda_clf.predict(X_scaled)

# Predizer probabilidades
y_proba = lda_clf.predict_proba(X_scaled)

# Acurácia
from sklearn.metrics import accuracy_score
acc = accuracy_score(y, y_pred)
print(f"Acurácia LDA: {acc:.2%}")

# Exemplo de predição
novo_texto_features = np.array([[0.5, 0.3, 0.2, 0.65, 0.60, 0.35, 3.8, 0.40, 0.03, 0.11]])
novo_texto_scaled = scaler.transform(novo_texto_features)

classe_pred = lda_clf.predict(novo_texto_scaled)[0]
proba_pred = lda_clf.predict_proba(novo_texto_scaled)[0]

print(f"Classe predita: {'LLM' if classe_pred == 1 else 'Humano'}")
print(f"Probabilidades: Humano={proba_pred[0]:.2%}, LLM={proba_pred[1]:.2%}")
```

### 3.6 Limitações do LDA

- ❌ **Linear:** Assume fronteira de decisão linear
- ❌ **Normalidade:** Assume features seguem distribuição normal (relaxável)
- ❌ **Homocedasticidade:** Assume covariâncias iguais entre classes
- ❌ **N° componentes:** Limitado a n_classes - 1 (1 componente para 2 classes)

**Vantagens:**
- ✅ **Supervisionado:** Usa labels para maximizar separação
- ✅ **Interpretável:** Coeficientes revelam importância de features
- ✅ **Rápido:** Solução fechada (não iterativa)

---

## 4. Regressão Logística

### 4.1 O Que É Regressão Logística?

**Regressão Logística** modela a **probabilidade** de um texto pertencer à classe LLM como função das características.

**Diferença de Regressão Linear:**

| Tipo | Saída | Range |
|------|-------|-------|
| Regressão Linear | Valor contínuo | (-∞, +∞) |
| Regressão Logística | Probabilidade | [0, 1] |

**Função logística (sigmoid):**

$$
P(y=1|x) = \frac{1}{1 + e^{-z}}
$$

Onde:
$$
z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p
$$

**Gráfico da sigmoid:**

```
P(y=1)
  1.0 |           ┌────────
      |         ╱
  0.5 |       ╱
      |     ╱
  0.0 | ───┘
      +─────────────────→ z
     -5   0   +5
```

### 4.2 Interpretação de Coeficientes

**Coeficiente β:**
- **β > 0:** Aumentar feature → aumenta probabilidade de LLM
- **β < 0:** Aumentar feature → diminui probabilidade de LLM (mais provável humano)
- **|β| grande:** Feature tem forte influência

**Odds ratio:**
$$
\text{OR} = e^\beta
$$

- OR = 2.0 → Aumentar feature em 1 unidade **duplica** as chances de ser LLM
- OR = 0.5 → Aumentar feature em 1 unidade **reduz pela metade** as chances

### 4.3 Implementação em Python

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Treinar Regressão Logística
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Predições
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]  # Probabilidade de LLM

# Avaliar
print("=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred, target_names=['Humano', 'LLM']))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("\n=== Matriz de Confusão ===")
print(f"                Predito")
print(f"              Humano  LLM")
print(f"Real Humano   {cm[0,0]:5d}  {cm[0,1]:4d}")
print(f"     LLM      {cm[1,0]:5d}  {cm[1,1]:4d}")

# AUC-ROC
auc = roc_auc_score(y_test, y_proba)
print(f"\nAUC-ROC: {auc:.4f}")
```

**Saída exemplo:**
```
=== Relatório de Classificação ===
              precision    recall  f1-score   support

      Humano       0.95      0.93      0.94       100
         LLM       0.93      0.95      0.94       100

    accuracy                           0.94       200
   macro avg       0.94      0.94      0.94       200
weighted avg       0.94      0.94      0.94       200

=== Matriz de Confusão ===
                Predito
              Humano  LLM
Real Humano      93    7
     LLM          5   95

AUC-ROC: 0.9703
```

### 4.4 Analisando Coeficientes

```python
# Extrair coeficientes
feature_names = ['sent_mean', 'sent_std', 'sent_burst', 'ttr', 'herdan_c',
                 'hapax_prop', 'char_entropy', 'func_word_ratio',
                 'first_person_ratio', 'bigram_repeat_ratio']

coefs = pd.DataFrame({
    'Feature': feature_names,
    'Coeficiente': logreg.coef_[0],
    'Odds_Ratio': np.exp(logreg.coef_[0])
})

# Ordenar por magnitude
coefs['Abs_Coef'] = coefs['Coeficiente'].abs()
coefs = coefs.sort_values('Abs_Coef', ascending=False)

print(coefs[['Feature', 'Coeficiente', 'Odds_Ratio']])

# Visualizar
plt.figure(figsize=(10, 6))
colors = ['red' if c < 0 else 'green' for c in coefs['Coeficiente']]
plt.barh(coefs['Feature'], coefs['Coeficiente'], color=colors, alpha=0.7)
plt.xlabel('Coeficiente (log-odds)', fontsize=12)
plt.title('Importância das Características - Regressão Logística', fontsize=14)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

**Interpretação exemplo:**

```
Feature               Coeficiente  Odds_Ratio
ttr                        +1.85        6.35
herdan_c                   +1.62        5.05
sent_mean                  +0.98        2.66
func_word_ratio            +0.75        2.12
char_entropy               +0.42        1.52
hapax_prop                 +0.38        1.46
bigram_repeat_ratio        -0.28        0.76
sent_std                   -0.85        0.43
sent_burst                 -1.12        0.33
first_person_ratio         -1.45        0.23
```

**Leitura:**
- **TTR:** Coef = +1.85, OR = 6.35
  - Aumentar TTR em 1 unidade (padronizada) → 6.35× mais chances de ser LLM
- **first_person_ratio:** Coef = -1.45, OR = 0.23
  - Aumentar pronomes 1ª pessoa em 1 unidade → 77% menos chances de ser LLM (mais humano)

### 4.5 Regularização (L1 e L2)

**Problema:** Overfitting com muitas features.

**Solução:** Adicionar penalidade aos coeficientes.

```python
# Ridge (L2): Penaliza coeficientes grandes
logreg_ridge = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)

# Lasso (L1): Força alguns coeficientes para zero (seleção de features)
logreg_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000)

# Treinar
logreg_ridge.fit(X_train, y_train)
logreg_lasso.fit(X_train, y_train)

# Comparar coeficientes
print("Coeficientes Ridge:", logreg_ridge.coef_[0])
print("Coeficientes Lasso:", logreg_lasso.coef_[0])
```

**Parâmetro C:**
- **C grande** (e.g., 10): Pouca regularização (pode overfittar)
- **C pequeno** (e.g., 0.1): Muita regularização (pode underfittar)

**Escolha de C:** Usar validação cruzada (Grid Search)

---

## 5. Random Forest

### 5.1 O Que É Random Forest?

**Random Forest** é um **ensemble** de múltiplas **árvores de decisão**.

**Árvore de decisão única:**

```
                     TTR < 0.65?
                    /           \
                  Sim            Não
                  /               \
         sent_mean < 21?      sent_mean < 24?
          /        \           /          \
       Humano    Humano      LLM         LLM
```

**Random Forest:**
- Treina **centenas** de árvores em subconjuntos aleatórios dos dados
- Cada árvore "vota" na classe final
- **Predição final:** Classe com mais votos

**Vantagens:**
- ✅ **Não-linear:** Captura interações complexas
- ✅ **Robusto:** Menos propenso a overfitting que árvore única
- ✅ **Importância de features:** Mede contribuição de cada feature
- ✅ **Sem suposições:** Não assume normalidade, homocedasticidade, etc.

### 5.2 Hiperparâmetros Importantes

| Parâmetro | O que faz | Valores típicos |
|-----------|-----------|-----------------|
| `n_estimators` | Número de árvores | 100-500 |
| `max_depth` | Profundidade máxima de cada árvore | 5-20 (ou None) |
| `min_samples_split` | Mín. amostras para dividir nó | 2-10 |
| `min_samples_leaf` | Mín. amostras em folha | 1-5 |
| `max_features` | N° features a considerar em cada split | 'sqrt', 'log2', None |

### 5.3 Implementação em Python

```python
from sklearn.ensemble import RandomForestClassifier

# Treinar Random Forest
rf = RandomForestClassifier(
    n_estimators=200,      # 200 árvores
    max_depth=10,          # Profundidade máxima 10
    min_samples_split=5,   # Mínimo 5 amostras para dividir
    min_samples_leaf=2,    # Mínimo 2 amostras em folha
    random_state=42,
    n_jobs=-1              # Usar todos os CPUs
)

rf.fit(X_train, y_train)

# Predições
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# Avaliar
from sklearn.metrics import accuracy_score, roc_auc_score
acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)

print(f"Acurácia Random Forest: {acc_rf:.2%}")
print(f"AUC-ROC: {auc_rf:.4f}")

print("\n=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred_rf, target_names=['Humano', 'LLM']))
```

### 5.4 Feature Importance

**Importância de feature no RF:** Média de redução de impureza (Gini) ao usar a feature.

```python
# Extrair importâncias
importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
})

# Ordenar
importances = importances.sort_values('Importance', ascending=False)

print(importances)

# Visualizar
plt.figure(figsize=(10, 6))
plt.barh(importances['Feature'], importances['Importance'], color='forestgreen', alpha=0.7)
plt.xlabel('Importância (Gini)', fontsize=12)
plt.title('Importância das Características - Random Forest', fontsize=14)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

**Saída exemplo:**
```
Feature               Importance
ttr                      0.245
herdan_c                 0.198
hapax_prop               0.152
sent_mean                0.118
func_word_ratio          0.095
char_entropy             0.072
sent_burst               0.058
sent_std                 0.045
first_person_ratio       0.012
bigram_repeat_ratio      0.005
```

**Interpretação:**
- **TTR** é a feature mais importante (24.5% do poder preditivo)
- **bigram_repeat_ratio** contribui pouco (0.5%)

### 5.5 Otimização de Hiperparâmetros (Grid Search)

```python
from sklearn.model_selection import GridSearchCV

# Definir grid de parâmetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search com validação cruzada 5-fold
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Melhores parâmetros
print("Melhores parâmetros:", grid_search.best_params_)
print(f"Melhor AUC (CV): {grid_search.best_score_:.4f}")

# Treinar modelo final com melhores parâmetros
rf_best = grid_search.best_estimator_
y_pred_best = rf_best.predict(X_test)
auc_best = roc_auc_score(y_test, rf_best.predict_proba(X_test)[:, 1])
print(f"AUC no teste: {auc_best:.4f}")
```

### 5.6 Limitações do Random Forest

- ❌ **Caixa-preta:** Difícil interpretar decisões individuais
- ❌ **Lento:** Treino e predição mais lentos que LR
- ❌ **Memória:** Requer muito espaço (centenas de árvores)
- ❌ **Extrapolação:** Não extrapola além do range de treino

**Vantagens:**
- ✅ **Alta acurácia:** Geralmente supera modelos lineares
- ✅ **Robusto:** Lida bem com outliers e dados ruidosos
- ✅ **Sem pré-processamento:** Não requer padronização

---

## 6. Comparação dos Métodos

### 6.1 Tabela Comparativa

| Método | Tipo | Supervisionado? | Linear? | Interpretável? | Acurácia típica |
|--------|------|----------------|---------|----------------|-----------------|
| **PCA** | Redução | Não | Sim | Moderado | N/A |
| **LDA** | Redução + Classificador | Sim | Sim | Alta | 85-92% |
| **Regressão Logística** | Classificador | Sim | Sim | Alta | 90-95% |
| **Random Forest** | Classificador | Sim | Não | Baixa | 93-98% |

### 6.2 Quando Usar Cada Método?

**PCA:**
- ✅ Visualizar dados em 2D/3D
- ✅ Explorar estrutura dos dados
- ✅ Reduzir dimensionalidade antes de classificar

**LDA:**
- ✅ Encontrar eixos discriminantes
- ✅ Interpretação científica (quais features separam classes?)
- ✅ Assumindo relações lineares

**Regressão Logística:**
- ✅ Modelo baseline (rápido, interpretável)
- ✅ Quando interpretação de coeficientes é importante
- ✅ Datasets pequenos/médios

**Random Forest:**
- ✅ Máxima acurácia preditiva
- ✅ Relações não-lineares entre features
- ✅ Importância de features

### 6.3 Código Comparativo

```python
from sklearn.metrics import roc_auc_score, accuracy_score
import time

# Modelos
models = {
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
}

# Treinar e avaliar
results = []
for name, model in models.items():
    # Tempo de treino
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        'Modelo': name,
        'Acurácia': acc,
        'AUC-ROC': auc,
        'Tempo (s)': train_time
    })

# Exibir
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
```

**Saída exemplo:**
```
                Modelo  Acurácia  AUC-ROC  Tempo (s)
                   LDA      0.89    0.921       0.02
  Logistic Regression      0.94    0.970       0.15
         Random Forest      0.97    0.985       2.34
```

---

## 7. Validação Cruzada

### 7.1 O Que É Validação Cruzada?

**Problema:** Treinar em TODO o dataset e testar no MESMO dataset → Overfitting!

**Solução:** Dividir dados em K folds, treinar em K-1, testar em 1, repetir K vezes.

**5-Fold Cross-Validation:**

```
Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]
Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]

Acurácia final = média das 5 acurácias
```

### 7.2 Implementação

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Definir estratégia de CV (estratificada = mantém proporção de classes)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Avaliar Regressão Logística
scores_lr = cross_val_score(
    LogisticRegression(max_iter=1000),
    X_scaled, y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

print(f"AUC-ROC por fold: {scores_lr}")
print(f"Média: {scores_lr.mean():.4f} ± {scores_lr.std():.4f}")

# Avaliar Random Forest
scores_rf = cross_val_score(
    RandomForestClassifier(n_estimators=200, random_state=42),
    X_scaled, y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

print(f"\nRandom Forest AUC: {scores_rf.mean():.4f} ± {scores_rf.std():.4f}")
```

**Saída:**
```
AUC-ROC por fold: [0.968 0.972 0.965 0.970 0.971]
Média: 0.9692 ± 0.0026

Random Forest AUC: 0.9845 ± 0.0018
```

**Interpretação:**
- Regressão Logística: **96.92% ± 0.26%**
- Random Forest: **98.45% ± 0.18%** (mais estável)

---

## 8. Métricas de Avaliação

### 8.1 Matriz de Confusão

```
                 Predito
              Humano  LLM
Real Humano     TN    FP
     LLM        FN    TP
```

- **TP** (True Positive): LLM corretamente identificado
- **TN** (True Negative): Humano corretamente identificado
- **FP** (False Positive): Humano classificado como LLM (Erro Tipo I)
- **FN** (False Negative): LLM classificado como humano (Erro Tipo II)

### 8.2 Métricas Derivadas

**Acurácia:**
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Precisão:**
$$
\text{Precision} = \frac{TP}{TP + FP}
$$
*"Dos que classifiquei como LLM, quantos eram realmente LLM?"*

**Recall (Sensibilidade):**
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
*"Dos LLMs reais, quantos consegui detectar?"*

**F1-Score:**
$$
F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### 8.3 Curva ROC e AUC

**ROC** (Receiver Operating Characteristic): Gráfico de TPR vs FPR em diferentes thresholds.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calcular ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plotar
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
```

**AUC** (Area Under Curve):
- **AUC = 1.0:** Classificador perfeito
- **AUC = 0.9-1.0:** Excelente
- **AUC = 0.8-0.9:** Bom
- **AUC = 0.7-0.8:** Razoável
- **AUC = 0.5:** Chance (inútil)

---

## 9. Próximo Passo

Continue para:
- **[GUIA_05_FUZZY.md](GUIA_05_FUZZY.md)** - Lógica Fuzzy, funções de pertinência e sistemas Takagi-Sugeno explicados em detalhe
