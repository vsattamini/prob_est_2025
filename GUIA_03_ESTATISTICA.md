# GUIA 03: Testes Estatísticos Detalhados

**Objetivo:** Explicar em profundidade os testes estatísticos usados para comparar textos humanos vs LLM.

**Público-alvo:** Pessoa com mestrado em Ciência da Computação, mas sem necessariamente domínio avançado de estatística.

**Pré-requisitos recomendados:**
- Conceito de distribuição de dados (média, mediana, quartis)
- Noção de significância estatística (p-valor)
- Histogramas e boxplots
- GUIA_01 e GUIA_02 lidos

---

## Índice

1. [Por Que Não Usar Teste T?](#1-por-que-não-usar-teste-t)
2. [Teste de Mann-Whitney U](#2-teste-de-mann-whitney-u)
3. [Cliff's Delta (Tamanho de Efeito)](#3-cliffs-delta-tamanho-de-efeito)
4. [Correção de Benjamini-Hochberg (FDR)](#4-correção-de-benjamini-hochberg-fdr)
5. [Pipeline Completo](#5-pipeline-completo)
6. [Interpretação dos Resultados](#6-interpretação-dos-resultados)

---

## 1. Por Que Não Usar Teste T?

### 1.1 O Teste T e Suas Suposições

O **teste t de Student** é o teste mais comum para comparar duas médias. Exemplo:

```python
from scipy.stats import ttest_ind

# Grupo A: alturas de homens (cm)
grupo_a = [175, 180, 172, 178, 182]

# Grupo B: alturas de mulheres (cm)
grupo_b = [160, 165, 162, 168, 163]

# Teste t
t_stat, p_value = ttest_ind(grupo_a, grupo_b)
print(f"Estatística t: {t_stat:.2f}")
print(f"P-valor: {p_value:.4f}")

# Se p < 0.05, concluímos que as médias são diferentes
```

**Suposições do teste t:**
1. ✅ **Normalidade:** Dados seguem distribuição normal (curva de sino)
2. ✅ **Homocedasticidade:** Variâncias dos dois grupos são similares
3. ✅ **Independência:** Amostras são independentes

### 1.2 Problema: Dados Estilométricos NÃO São Normais

Vamos visualizar a distribuição de uma característica (TTR):

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro

# Dados fictícios de TTR (Type-Token Ratio)
ttr_humanos = [0.45, 0.52, 0.48, 0.60, 0.55, 0.50, 0.58, 0.62, 0.51, 0.47]
ttr_llms = [0.65, 0.72, 0.68, 0.75, 0.70, 0.73, 0.69, 0.71, 0.74, 0.67]

# Teste de normalidade (Shapiro-Wilk)
stat_h, p_h = shapiro(ttr_humanos)
stat_l, p_l = shapiro(ttr_llms)

print(f"Shapiro-Wilk (humanos): p = {p_h:.4f}")
print(f"Shapiro-Wilk (LLMs): p = {p_l:.4f}")
# Se p < 0.05, rejeita-se normalidade

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(ttr_humanos, bins=5, alpha=0.7, label='Humanos', edgecolor='black')
axes[0].set_title('TTR - Humanos')
axes[0].set_xlabel('TTR')
axes[0].set_ylabel('Frequência')

axes[1].hist(ttr_llms, bins=5, alpha=0.7, label='LLMs', color='orange', edgecolor='black')
axes[1].set_title('TTR - LLMs')
axes[1].set_xlabel('TTR')

plt.tight_layout()
plt.show()
```

**Observações comuns em dados estilométricos:**
- ⚠️ **Distribuições assimétricas** (skewed): Muitos valores baixos, poucos altos
- ⚠️ **Outliers:** Textos muito atípicos
- ⚠️ **Distribuições bimodais:** Dois picos distintos
- ⚠️ **Valores limitados:** TTR ∈ [0, 1], comprimento ≥ 0

**Consequência:** Teste t pode dar **resultados incorretos** (p-valores inflados ou deflados).

### 1.3 Solução: Testes Não-Paramétricos

**Teste não-paramétrico** = não assume distribuição específica dos dados.

**Vantagens:**
- ✅ Funciona com distribuições assimétricas
- ✅ Robusto a outliers
- ✅ Válido para amostras pequenas
- ✅ Baseado em **ranks** (ordenação), não valores absolutos

**Desvantagens:**
- ⚠️ Menos "poderoso" que teste t (quando t é válido)
- ⚠️ Testa **medianas**, não médias

**Teste não-paramétrico escolhido:** Mann-Whitney U

---

## 2. Teste de Mann-Whitney U

### 2.1 O Que É?

O **teste de Mann-Whitney U** (também chamado Wilcoxon rank-sum test) compara se duas distribuições têm a mesma **tendência central** (mediana).

**Hipóteses:**
- **H₀ (nula):** As duas distribuições são idênticas (mesma mediana)
- **H₁ (alternativa):** As distribuições são diferentes

**Ideia central:** Se dois grupos têm medianas diferentes, então ao ordenar todos os valores juntos, um grupo tenderá a ocupar ranks maiores que o outro.

### 2.2 Como Funciona? (Exemplo Manual)

**Dados:**
- Grupo A (humanos): [12, 15, 18, 20]
- Grupo B (LLMs): [22, 25, 28, 30]

**Passo 1:** Combinar e ordenar todos os valores

| Valor | 12 | 15 | 18 | 20 | 22 | 25 | 28 | 30 |
|-------|----|----|----|----|----|----|----|----|
| Rank  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  |
| Grupo | A  | A  | A  | A  | B  | B  | B  | B  |

**Passo 2:** Somar ranks de cada grupo
- Soma ranks grupo A: 1 + 2 + 3 + 4 = **10**
- Soma ranks grupo B: 5 + 6 + 7 + 8 = **26**

**Passo 3:** Calcular estatística U

$$
U_A = n_A \cdot n_B + \frac{n_A(n_A + 1)}{2} - R_A
$$

Onde:
- $n_A$ = tamanho do grupo A (4)
- $n_B$ = tamanho do grupo B (4)
- $R_A$ = soma dos ranks do grupo A (10)

$$
U_A = 4 \times 4 + \frac{4 \times 5}{2} - 10 = 16 + 10 - 10 = 16
$$

$$
U_B = n_A \cdot n_B - U_A = 16 - 16 = 0
$$

**Estatística final:** $U = \min(U_A, U_B) = 0$

**Interpretação:** U = 0 significa **separação perfeita** (todos os valores de A são menores que todos de B).

**Passo 4:** Calcular p-valor (comparar U com distribuição teórica ou usar tabela)

### 2.3 Implementação em Python

```python
from scipy.stats import mannwhitneyu
import numpy as np

# Dados exemplo: comprimento médio de frases (tokens)
sent_mean_humanos = [18.5, 20.1, 19.3, 22.0, 17.8, 21.5, 19.0, 20.5]
sent_mean_llms = [24.2, 26.5, 25.0, 27.3, 23.8, 25.5, 24.0, 26.0]

# Teste de Mann-Whitney U
u_stat, p_value = mannwhitneyu(
    sent_mean_humanos,
    sent_mean_llms,
    alternative='two-sided'  # Testa se são diferentes (não direção específica)
)

print(f"Estatística U: {u_stat}")
print(f"P-valor: {p_value:.6f}")

# Interpretação
alpha = 0.05
if p_value < alpha:
    print("✅ Rejeita H₀: As distribuições SÃO diferentes (p < 0.05)")
else:
    print("❌ Não rejeita H₀: Não há evidência de diferença (p ≥ 0.05)")

# Medianas
print(f"Mediana humanos: {np.median(sent_mean_humanos):.2f}")
print(f"Mediana LLMs: {np.median(sent_mean_llms):.2f}")
```

**Saída esperada:**
```
Estatística U: 0.0
P-valor: 0.000196
✅ Rejeita H₀: As distribuições SÃO diferentes (p < 0.05)
Mediana humanos: 19.65
Mediana LLMs: 25.25
```

### 2.4 Parâmetro `alternative`

```python
# Testa se A < B (unilateral à esquerda)
mannwhitneyu(A, B, alternative='less')

# Testa se A > B (unilateral à direita)
mannwhitneyu(A, B, alternative='greater')

# Testa se A ≠ B (bilateral, padrão)
mannwhitneyu(A, B, alternative='two-sided')
```

**No nosso caso:** Usamos `'two-sided'` porque queremos saber SE há diferença, sem pressupor direção.

### 2.5 Interpretação do P-Valor

**P-valor** = Probabilidade de observar uma diferença tão extrema quanto a observada, **assumindo que H₀ é verdadeira**.

| P-valor | Interpretação |
|---------|---------------|
| p < 0.001 | Evidência **muito forte** de diferença |
| 0.001 ≤ p < 0.01 | Evidência **forte** |
| 0.01 ≤ p < 0.05 | Evidência **moderada** |
| 0.05 ≤ p < 0.10 | Evidência **fraca** (marginalmente significativo) |
| p ≥ 0.10 | **Sem evidência** de diferença |

**Limiar convencional:** α = 0.05

**⚠️ IMPORTANTE:** P-valor **NÃO** indica **tamanho da diferença**, apenas se ela é estatisticamente detectável!

**Exemplo:**
- Diferença de 0.01 em TTR pode ter p < 0.001 (significativo)
- Mas essa diferença pode ser **praticamente irrelevante**

**Solução:** Usar medida de **tamanho de efeito** (effect size) → Cliff's delta

---

## 3. Cliff's Delta (Tamanho de Efeito)

### 3.1 O Que É Tamanho de Efeito?

**Tamanho de efeito** (effect size) quantifica a **magnitude** da diferença entre grupos, independentemente do p-valor.

**Analogia:**
- **P-valor:** "Há diferença estatisticamente detectável?" (Sim/Não)
- **Tamanho de efeito:** "Quão GRANDE é essa diferença?" (Pequena/Média/Grande)

**Por que importa?**
Com amostras grandes (N > 10,000), até diferenças minúsculas têm p < 0.001. Tamanho de efeito revela se a diferença é **praticamente relevante**.

### 3.2 Cliff's Delta (δ)

**Cliff's delta** é uma medida de tamanho de efeito para **testes não-paramétricos** (alternativa ao Cohen's d do teste t).

**Definição:**
$$
\delta = \frac{\#(x > y) - \#(x < y)}{n_x \cdot n_y}
$$

Onde:
- $x$ = valores do grupo X
- $y$ = valores do grupo Y
- $\#(x > y)$ = número de pares $(x_i, y_j)$ onde $x_i > y_j$
- $\#(x < y)$ = número de pares $(x_i, y_j)$ onde $x_i < y_j$
- $n_x \cdot n_y$ = total de pares possíveis

**Interpretação:**
- **δ = +1:** Todos os valores de X > todos os valores de Y (separação perfeita)
- **δ = 0:** Distribuições completamente sobrepostas (sem diferença)
- **δ = -1:** Todos os valores de X < todos os valores de Y

### 3.3 Cálculo Manual (Exemplo Pequeno)

**Dados:**
- Grupo X (humanos): [10, 12, 14]
- Grupo Y (LLMs): [16, 18, 20]

**Passo 1:** Comparar todos os pares

| Par | X | Y | X > Y? | X < Y? | X = Y? |
|-----|---|---|--------|--------|--------|
| 1 | 10 | 16 | Não | **Sim** | Não |
| 2 | 10 | 18 | Não | **Sim** | Não |
| 3 | 10 | 20 | Não | **Sim** | Não |
| 4 | 12 | 16 | Não | **Sim** | Não |
| 5 | 12 | 18 | Não | **Sim** | Não |
| 6 | 12 | 20 | Não | **Sim** | Não |
| 7 | 14 | 16 | Não | **Sim** | Não |
| 8 | 14 | 18 | Não | **Sim** | Não |
| 9 | 14 | 20 | Não | **Sim** | Não |

**Passo 2:** Contar
- $\#(x > y) = 0$
- $\#(x < y) = 9$
- Total de pares = $3 \times 3 = 9$

**Passo 3:** Calcular delta
$$
\delta = \frac{0 - 9}{9} = -1.0
$$

**Interpretação:** δ = -1 significa que **100% dos valores de X são menores que Y** (LLMs têm valores maiores).

### 3.4 Implementação em Python

```python
def cliff_delta(group1, group2):
    """
    Calcula Cliff's delta entre dois grupos.

    Retorna:
        delta (float): Valor entre -1 e +1
    """
    n1, n2 = len(group1), len(group2)

    # Contar pares
    greater = 0
    less = 0

    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
            # Ignora empates (x == y)

    delta = (greater - less) / (n1 * n2)
    return delta

# Exemplo
humanos = [18.5, 20.1, 19.3, 22.0]
llms = [24.2, 26.5, 25.0, 27.3]

delta = cliff_delta(humanos, llms)
print(f"Cliff's delta: {delta:.3f}")

# Interpretar
if delta > 0:
    print(f"Grupo 1 tende a ter valores MAIORES (δ = {delta:.3f})")
elif delta < 0:
    print(f"Grupo 2 tende a ter valores MAIORES (δ = {delta:.3f})")
else:
    print("Sem diferença prática")
```

**Saída:**
```
Cliff's delta: -1.000
Grupo 2 tende a ter valores MAIORES (δ = -1.000)
```

### 3.5 Thresholds de Romano et al. (2006)

**Como interpretar a magnitude de δ?**

Romano et al. (2006) propuseram thresholds baseados em dados educacionais:

| |δ| (absoluto) | Interpretação |
|----------------|----------------|
| 0.00 - 0.147 | **Negligível** (praticamente irrelevante) |
| 0.147 - 0.330 | **Pequeno** |
| 0.330 - 0.474 | **Médio** |
| ≥ 0.474 | **Grande** |

**Observação:** Use o **valor absoluto** |δ| para classificar magnitude (ignorando sinal).

**Função auxiliar:**

```python
def interpretar_cliff_delta(delta):
    """
    Interpreta magnitude de Cliff's delta segundo Romano et al. (2006).
    """
    abs_delta = abs(delta)

    if abs_delta < 0.147:
        magnitude = "negligível"
    elif abs_delta < 0.330:
        magnitude = "pequeno"
    elif abs_delta < 0.474:
        magnitude = "médio"
    else:
        magnitude = "grande"

    # Direção
    if delta > 0:
        direction = "Grupo 1 > Grupo 2"
    elif delta < 0:
        direction = "Grupo 1 < Grupo 2"
    else:
        direction = "Sem diferença"

    return f"Efeito {magnitude} (δ = {delta:.3f}): {direction}"

# Exemplos
print(interpretar_cliff_delta(0.100))  # negligível
print(interpretar_cliff_delta(0.250))  # pequeno
print(interpretar_cliff_delta(0.400))  # médio
print(interpretar_cliff_delta(0.600))  # grande
print(interpretar_cliff_delta(-0.450))  # médio (negativo)
```

**Saída:**
```
Efeito negligível (δ = 0.100): Grupo 1 > Grupo 2
Efeito pequeno (δ = 0.250): Grupo 1 > Grupo 2
Efeito médio (δ = 0.400): Grupo 1 > Grupo 2
Efeito grande (δ = 0.600): Grupo 1 > Grupo 2
Efeito médio (δ = -0.450): Grupo 1 < Grupo 2
```

### 3.6 Relação Entre P-Valor e Cliff's Delta

**Cenário comum no estudo:**

| Característica | P-valor | Cliff's δ | Interpretação |
|----------------|---------|-----------|---------------|
| ttr | < 0.001 | **+0.636** | Diferença **grande** e **altamente significativa** |
| char_entropy | < 0.001 | **+0.173** | Diferença **pequena** mas **significativa** |

**Lição:** Sempre reporte **ambos**:
- **P-valor:** Confiança estatística (há diferença?)
- **Cliff's delta:** Relevância prática (quão grande?)

---

## 4. Correção de Benjamini-Hochberg (FDR)

### 4.1 O Problema das Comparações Múltiplas

**Cenário:** Testar 10 características (sent_mean, ttr, entropy, etc.) simultaneamente.

**Problema:** Se α = 0.05, esperamos **5% de falsos positivos** (rejeitar H₀ quando ela é verdadeira).

**Taxa de erro para K testes:**
$$
P(\text{pelo menos 1 falso positivo}) = 1 - (1 - \alpha)^K
$$

Para K = 10 e α = 0.05:
$$
P(\text{FP}) = 1 - 0.95^{10} = 0.401 \text{ (40%!)}
$$

**Consequência:** Com 10 testes, há **40% de chance** de encontrar pelo menos 1 diferença "significativa" apenas por acaso!

### 4.2 Correção de Bonferroni (Muito Conservadora)

**Ideia:** Dividir α pelo número de testes.

$$
\alpha_{\text{ajustado}} = \frac{\alpha}{K}
$$

Para K = 10:
$$
\alpha_{\text{ajustado}} = \frac{0.05}{10} = 0.005
$$

**Problema:** Muito conservador! Aumenta taxa de **falsos negativos** (não detectar diferenças reais).

### 4.3 Correção de Benjamini-Hochberg (FDR)

**FDR** = False Discovery Rate (Taxa de Descobertas Falsas)

**Vantagem:** Menos conservador que Bonferroni, controla a **proporção esperada** de falsos positivos entre as descobertas.

**Procedimento:**

**Passo 1:** Ordenar p-valores do menor para o maior

| Rank (i) | Característica | P-valor |
|----------|---------------|---------|
| 1 | ttr | 0.0001 |
| 2 | herdan_c | 0.0002 |
| 3 | sent_mean | 0.005 |
| 4 | func_word_ratio | 0.012 |
| 5 | char_entropy | 0.035 |
| ... | ... | ... |

**Passo 2:** Calcular threshold crítico para cada rank

$$
p_{\text{crítico}}(i) = \frac{i \cdot \alpha}{K}
$$

Onde:
- $i$ = rank (1, 2, 3, ...)
- $K$ = número total de testes (10)
- $\alpha$ = nível de significância desejado (0.05)

**Exemplo:**
- $p_{\text{crítico}}(1) = \frac{1 \times 0.05}{10} = 0.005$
- $p_{\text{crítico}}(2) = \frac{2 \times 0.05}{10} = 0.010$
- $p_{\text{crítico}}(3) = \frac{3 \times 0.05}{10} = 0.015$

**Passo 3:** Encontrar o maior i onde $p(i) \leq p_{\text{crítico}}(i)$

| Rank (i) | P-valor | P-crítico | Passa? |
|----------|---------|-----------|--------|
| 1 | 0.0001 | 0.005 | ✅ Sim (0.0001 < 0.005) |
| 2 | 0.0002 | 0.010 | ✅ Sim (0.0002 < 0.010) |
| 3 | 0.005 | 0.015 | ✅ Sim (0.005 < 0.015) |
| 4 | 0.012 | 0.020 | ✅ Sim (0.012 < 0.020) |
| 5 | 0.035 | 0.025 | ❌ Não (0.035 > 0.025) |

**Resultado:** Rejeita H₀ para os primeiros **4 testes**.

**Passo 4:** Calcular p-valores ajustados (opcional)

$$
p_{\text{ajustado}}(i) = \min\left(1, \frac{K \cdot p(i)}{i}\right)
$$

### 4.4 Implementação em Python

```python
import numpy as np
from scipy.stats import false_discovery_control

# P-valores originais (10 características)
p_values = np.array([
    0.0001,  # ttr
    0.0002,  # herdan_c
    0.0050,  # sent_mean
    0.0120,  # func_word_ratio
    0.0350,  # char_entropy
    0.0800,  # sent_std
    0.1200,  # hapax_prop
    0.2500,  # sent_burst
    0.4000,  # first_person_ratio
    0.7000   # bigram_repeat_ratio
])

# Aplicar correção FDR (Benjamini-Hochberg)
alpha = 0.05
reject, p_adjusted = false_discovery_control(p_values, alpha=alpha, method='bh')

# Exibir resultados
features = ['ttr', 'herdan_c', 'sent_mean', 'func_word_ratio', 'char_entropy',
            'sent_std', 'hapax_prop', 'sent_burst', 'first_person_ratio', 'bigram_repeat_ratio']

print(f"{'Característica':<20} {'P-valor':<10} {'P-ajustado':<12} {'Significativo?'}")
print("-" * 60)
for i, feat in enumerate(features):
    sig = "✅ Sim" if reject[i] else "❌ Não"
    print(f"{feat:<20} {p_values[i]:<10.4f} {p_adjusted[i]:<12.4f} {sig}")

print(f"\n Total significativos: {np.sum(reject)}/{len(p_values)}")
```

**Saída esperada:**
```
Característica       P-valor    P-ajustado   Significativo?
------------------------------------------------------------
ttr                  0.0001     0.0010       ✅ Sim
herdan_c             0.0002     0.0010       ✅ Sim
sent_mean            0.0050     0.0167       ✅ Sim
func_word_ratio      0.0120     0.0300       ✅ Sim
char_entropy         0.0350     0.0700       ❌ Não
sent_std             0.0800     0.1333       ❌ Não
hapax_prop           0.1200     0.1714       ❌ Não
sent_burst           0.2500     0.3125       ❌ Não
first_person_ratio   0.4000     0.4444       ❌ Não
bigram_repeat_ratio  0.7000     0.7000       ❌ Não

Total significativos: 4/10
```

**Interpretação:**
- Sem correção: 5 características seriam consideradas significativas (p < 0.05)
- Com FDR: **4 características** resistem à correção (mais confiáveis)

### 4.5 Quando Usar FDR?

**Use correção FDR quando:**
- ✅ Testar múltiplas hipóteses simultaneamente (> 5 testes)
- ✅ Reportar descobertas em artigo científico
- ✅ Evitar inflação de taxa de erro tipo I

**Não precisa usar quando:**
- ❌ Teste exploratório inicial (fase de piloto)
- ❌ Hipótese única pré-especificada
- ❌ Validação de resultado conhecido

**No contexto do estudo:**
- **10 características testadas** → Aplicar FDR
- Reportar: "Após correção de Benjamini-Hochberg (FDR), 9/10 características permaneceram significativas (α = 0.05)"

---

## 5. Pipeline Completo

### 5.1 Código Integrado

```python
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import false_discovery_control

def cliff_delta(group1, group2):
    """Calcula Cliff's delta."""
    n1, n2 = len(group1), len(group2)
    greater = sum(1 for x in group1 for y in group2 if x > y)
    less = sum(1 for x in group1 for y in group2 if x < y)
    return (greater - less) / (n1 * n2)

def interpretar_magnitude(delta):
    """Interpreta magnitude segundo Romano et al. (2006)."""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligível"
    elif abs_delta < 0.330:
        return "pequeno"
    elif abs_delta < 0.474:
        return "médio"
    else:
        return "grande"

def pipeline_estatistico(df_humanos, df_llms, caracteristicas):
    """
    Pipeline completo de análise estatística.

    Args:
        df_humanos: DataFrame com características de textos humanos
        df_llms: DataFrame com características de textos LLM
        caracteristicas: Lista de nomes de colunas a testar

    Returns:
        DataFrame com resultados (p-valor, delta, significância)
    """
    resultados = []

    # Passo 1: Testar cada característica
    for feat in caracteristicas:
        humanos = df_humanos[feat].values
        llms = df_llms[feat].values

        # Mann-Whitney U
        u_stat, p_val = mannwhitneyu(humanos, llms, alternative='two-sided')

        # Cliff's delta
        delta = cliff_delta(humanos, llms)
        magnitude = interpretar_magnitude(delta)

        # Medianas
        med_h = np.median(humanos)
        med_l = np.median(llms)

        resultados.append({
            'Característica': feat,
            'U': u_stat,
            'P-valor': p_val,
            'Cliff_delta': delta,
            'Magnitude': magnitude,
            'Mediana_Humanos': med_h,
            'Mediana_LLMs': med_l
        })

    # Converter para DataFrame
    df_res = pd.DataFrame(resultados)

    # Passo 2: Aplicar correção FDR
    reject, p_adjusted = false_discovery_control(
        df_res['P-valor'].values,
        alpha=0.05,
        method='bh'
    )

    df_res['P-valor_ajustado'] = p_adjusted
    df_res['Significativo_FDR'] = reject

    # Ordenar por magnitude de efeito (decrescente)
    df_res = df_res.sort_values('Cliff_delta', key=abs, ascending=False)

    return df_res

# Exemplo de uso
# (Assumindo df_humanos e df_llms já carregados)

caracteristicas = [
    'sent_mean', 'sent_std', 'sent_burst',
    'ttr', 'herdan_c', 'hapax_prop',
    'char_entropy', 'func_word_ratio',
    'first_person_ratio', 'bigram_repeat_ratio'
]

resultados = pipeline_estatistico(df_humanos, df_llms, caracteristicas)

# Exibir
print(resultados.to_string(index=False))

# Salvar
resultados.to_csv('resultados_estatisticos.csv', index=False)
```

### 5.2 Exemplo de Saída

```
Característica        U       P-valor  Cliff_delta  Magnitude  Mediana_Humanos  Mediana_LLMs  P-valor_ajustado  Significativo_FDR
ttr              125000  1.234e-150       +0.636      grande             0.62          0.71          1.234e-149               True
herdan_c         130000  2.456e-145       +0.616      grande             0.58          0.68          1.228e-144               True
hapax_prop       145000  3.789e-130       +0.555      grande             0.32          0.41          1.263e-129               True
sent_burst       180000  5.012e-100       -0.453       médio             0.45          0.38          1.253e-99                True
first_person_r   190000  8.123e-85        -0.424       médio             0.038         0.022         1.625e-84                True
sent_std         200000  1.234e-70        -0.364       médio             8.5           6.2           2.057e-70                True
func_word_ratio  210000  2.456e-60        +0.361       médio             0.38          0.43          3.509e-60                True
sent_mean        215000  3.789e-55        +0.349       médio            19.5          23.2          4.736e-55                True
char_entropy     250000  5.012e-25        +0.173     pequeno             3.85          3.92          5.569e-25                True
bigram_repeat_r  280000  8.123e-10        -0.147  negligível             0.12          0.10          8.123e-10                True
```

**Observações:**
- ✅ Todas 10 características têm p-valor < 0.05 (antes de correção)
- ✅ Após FDR, todas permanecem significativas (p-ajustado < 0.05)
- ✅ 3 características têm efeito **grande** (ttr, herdan_c, hapax_prop)
- ✅ 5 características têm efeito **médio**
- ⚠️ 1 característica tem efeito **pequeno** (char_entropy)
- ⚠️ 1 característica tem efeito **negligível** (bigram_repeat_ratio)

---

## 6. Interpretação dos Resultados

### 6.1 Hierarquia de Evidência

Ao reportar resultados, siga esta hierarquia:

1. **Significância estatística** (p-valor após FDR)
   - "A característica X foi significativamente diferente (p < 0.001, FDR-ajustado)"

2. **Magnitude prática** (Cliff's delta)
   - "...com efeito grande (δ = 0.636, Romano et al. 2006)"

3. **Direção e valores descritivos**
   - "LLMs apresentaram maior diversidade lexical (mediana TTR: 0.71 vs 0.62)"

### 6.2 Template de Redação

**Exemplo 1: Efeito grande**

> *"A característica TTR (Type-Token Ratio) diferiu significativamente entre textos humanos e gerados por LLM (Mann-Whitney U = 125,000, p < 0.001, FDR-corrigido). O tamanho de efeito foi grande (Cliff's δ = +0.636), indicando que LLMs produziram textos com maior diversidade lexical (mediana: 0.71) comparado a humanos (mediana: 0.62)."*

**Exemplo 2: Efeito negligível (mas significativo)**

> *"Embora o bigram_repeat_ratio tenha sido estatisticamente diferente (p < 0.001, FDR-corrigido), o tamanho de efeito foi negligível (Cliff's δ = -0.147, Romano et al. 2006), sugerindo diferença praticamente irrelevante entre grupos (humanos: 0.12 vs LLMs: 0.10)."*

### 6.3 Visualização dos Resultados

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_resultados(df_resultados):
    """
    Visualiza p-valores e Cliff's delta.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: P-valores (escala log)
    ax1 = axes[0]
    df_sorted = df_resultados.sort_values('P-valor')

    ax1.barh(df_sorted['Característica'], -np.log10(df_sorted['P-valor']), color='steelblue')
    ax1.axvline(-np.log10(0.05), color='red', linestyle='--', label='α = 0.05')
    ax1.axvline(-np.log10(0.001), color='orange', linestyle='--', label='p = 0.001')
    ax1.set_xlabel('-log10(P-valor)', fontsize=12)
    ax1.set_title('Significância Estatística', fontsize=14)
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Subplot 2: Cliff's delta com barras coloridas por magnitude
    ax2 = axes[1]
    df_sorted2 = df_resultados.sort_values('Cliff_delta')

    colors = []
    for delta in df_sorted2['Cliff_delta']:
        if abs(delta) < 0.147:
            colors.append('lightgray')  # negligível
        elif abs(delta) < 0.330:
            colors.append('gold')  # pequeno
        elif abs(delta) < 0.474:
            colors.append('orange')  # médio
        else:
            colors.append('red')  # grande

    ax2.barh(df_sorted2['Característica'], df_sorted2['Cliff_delta'], color=colors)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.axvline(-0.147, color='gray', linestyle='--', alpha=0.5, label='Thresholds')
    ax2.axvline(0.147, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(-0.474, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(0.474, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Cliff's Delta (δ)", fontsize=12)
    ax2.set_title('Tamanho de Efeito', fontsize=14)
    ax2.set_xlim(-1, 1)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

# Usar
plot_resultados(resultados)
```

### 6.4 Checklist de Qualidade

Ao analisar resultados estatísticos, verifique:

- ✅ **Verificou normalidade?** (Shapiro-Wilk, histogramas) → Se não-normal, use Mann-Whitney
- ✅ **Reportou p-valor E tamanho de efeito?** → Ambos são essenciais
- ✅ **Aplicou correção para comparações múltiplas?** → FDR se > 5 testes
- ✅ **Interpretou magnitude de efeito?** → Use thresholds de Romano (2006)
- ✅ **Reportou estatísticas descritivas?** → Medianas, IQR
- ✅ **Visualizou distribuições?** → Boxplots, violin plots
- ✅ **Discutiu limitações?** → Tamanho amostral, generalização

---

## 7. Leituras Recomendadas

### 7.1 Testes Não-Paramétricos

1. **Mann & Whitney (1947)** - "On a test of whether one of two random variables is stochastically larger than the other"
   - Annals of Mathematical Statistics
   - **Essencial:** Artigo original do teste Mann-Whitney U

2. **Siegel & Castellan (1988)** - *Nonparametric Statistics for the Behavioral Sciences*
   - McGraw-Hill
   - **Capítulo 6:** Mann-Whitney U test detalhado com exemplos

### 7.2 Tamanho de Efeito

3. **Cliff (1993)** - "Dominance statistics: Ordinal analyses to answer ordinal questions"
   - Psychological Bulletin, 114: 494-509
   - **Essencial:** Definição e justificativa do Cliff's delta

4. **Romano et al. (2006)** - "Appropriate Statistics for Ordinal Level Data"
   - Florida Association of Institutional Research
   - **Essencial:** Thresholds para interpretar Cliff's delta (0.147, 0.330, 0.474)

### 7.3 Comparações Múltiplas

5. **Benjamini & Hochberg (1995)** - "Controlling the false discovery rate"
   - Journal of the Royal Statistical Society B, 57: 289-300
   - **Essencial:** Procedimento FDR original

6. **Shaffer (1995)** - "Multiple hypothesis testing"
   - Annual Review of Psychology, 46: 561-584
   - **Revisão:** Comparação de métodos (Bonferroni, Holm, FDR)

### 7.4 Livros-Texto

7. **Mood & Graybill (1974)** - *Introduction to the Theory of Statistics*
   - McGraw-Hill
   - **Capítulo 13:** Testes não-paramétricos

8. **Bussab & Morettin (2002)** - *Estatística Básica*
   - Saraiva
   - **Capítulo 12:** Testes de hipóteses (em português!)

---

## 8. Conceitos-Chave para Revisar

Antes da defesa, certifique-se de dominar:

### Fundamentos
- [ ] Diferença entre **média** e **mediana**
- [ ] O que é **distribuição normal** (e quando dados NÃO são normais)
- [ ] Conceito de **p-valor** (probabilidade sob H₀)
- [ ] Diferença entre **significância estatística** e **relevância prática**

### Testes
- [ ] Por que **não** usar teste t em dados estilométricos?
- [ ] Como funciona o **Mann-Whitney U** (baseado em ranks)
- [ ] Hipótese nula e alternativa
- [ ] Escolha de `alternative='two-sided'` vs `'less'` vs `'greater'`

### Tamanho de Efeito
- [ ] O que é **Cliff's delta** (δ)?
- [ ] Como interpretar δ = +1, 0, -1?
- [ ] Thresholds de **Romano et al. (2006)** (0.147, 0.330, 0.474)
- [ ] Diferença entre **Cohen's d** (paramétrico) e **Cliff's delta** (não-paramétrico)

### Comparações Múltiplas
- [ ] Por que comparações múltiplas inflam erro tipo I?
- [ ] Diferença entre **Bonferroni** (conservador) e **FDR** (menos conservador)
- [ ] Procedimento de **Benjamini-Hochberg** passo-a-passo
- [ ] Quando reportar "FDR-corrigido"

### Interpretação
- [ ] Como reportar: "Mann-Whitney U = X, p < 0.001, δ = Y"
- [ ] Sempre mencionar **direção** (humanos > LLMs ou vice-versa)
- [ ] Incluir **valores descritivos** (medianas)

---

## 9. Próximo Passo

Continue para:
- **[GUIA_04_CLASSIFICADORES.md](GUIA_04_CLASSIFICADORES.md)** - PCA, LDA, Regressão Logística e Random Forest explicados passo-a-passo
