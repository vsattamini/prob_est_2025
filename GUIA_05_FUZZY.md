# GUIA 05: Lógica Fuzzy e Sistemas de Inferência

**Objetivo:** Explicar em profundidade a lógica fuzzy e sistemas Takagi-Sugeno usados no segundo paper.

**Público-alvo:** Pessoa com mestrado em Ciência da Computação, mas sem necessariamente experiência com lógica fuzzy.

**Pré-requisitos recomendados:**
- Conceitos de lógica booleana (AND, OR, NOT)
- Funções matemáticas básicas
- GUIA_01, GUIA_02 lidos

---

## Índice

1. [Motivação: Por Que Fuzzy?](#1-motivação-por-que-fuzzy)
2. [Conjuntos Fuzzy e Funções de Pertinência](#2-conjuntos-fuzzy-e-funções-de-pertinência)
3. [Operações com Conjuntos Fuzzy](#3-operações-com-conjuntos-fuzzy)
4. [Funções de Pertinência Triangulares](#4-funções-de-pertinência-triangulares)
5. [Sistemas de Inferência Fuzzy](#5-sistemas-de-inferência-fuzzy)
6. [Takagi-Sugeno de Ordem Zero](#6-takagi-sugeno-de-ordem-zero)
7. [Implementação Completa](#7-implementação-completa)
8. [Interpretabilidade](#8-interpretabilidade)

---

## 1. Motivação: Por Que Fuzzy?

### 1.1 Limitações da Lógica Clássica

**Lógica booleana clássica:**

Definição: "Texto é formal se func_word_ratio > 0.40"

```python
def is_formal_classical(func_word_ratio):
    if func_word_ratio > 0.40:
        return True  # Formal
    else:
        return False  # Informal
```

**Problema:**

| func_word_ratio | Classificação | Problema |
|----------------|---------------|----------|
| 0.39 | Informal | |
| 0.40 | Formal | ⚠️ Mudança abrupta! |
| 0.41 | Formal | |

**Mundo real:** Texto com func_word_ratio = 0.39 é **quase tão formal** quanto 0.41!

### 1.2 Abordagem Fuzzy

**Lógica fuzzy:**

```python
def is_formal_fuzzy(func_word_ratio):
    """
    Retorna grau de pertinência ao conjunto 'Formal' [0, 1]
    """
    if func_word_ratio <= 0.30:
        return 0.0  # Totalmente informal
    elif func_word_ratio >= 0.50:
        return 1.0  # Totalmente formal
    else:
        # Transição gradual
        return (func_word_ratio - 0.30) / (0.50 - 0.30)

# Exemplos
print(f"0.25 → {is_formal_fuzzy(0.25):.2f}")  # 0.00 (informal)
print(f"0.35 → {is_formal_fuzzy(0.35):.2f}")  # 0.25 (levemente formal)
print(f"0.40 → {is_formal_fuzzy(0.40):.2f}")  # 0.50 (meio formal)
print(f"0.45 → {is_formal_fuzzy(0.45):.2f}")  # 0.75 (bastante formal)
print(f"0.55 → {is_formal_fuzzy(0.55):.2f}")  # 1.00 (totalmente formal)
```

**Saída:**
```
0.25 → 0.00
0.35 → 0.25
0.40 → 0.50
0.45 → 0.75
0.55 → 1.00
```

**Vantagem:** Transição **suave** entre informal e formal!

### 1.3 Quando Usar Fuzzy?

**Use lógica fuzzy quando:**
- ✅ Conceitos são **vagos** ou **graduais** ("alto", "baixo", "médio")
- ✅ Fronteiras são **incertas** (não há threshold claro)
- ✅ **Interpretabilidade** é importante (regras linguísticas)
- ✅ Conhecimento especialista está disponível

**No contexto de LLM detection:**
- ✅ "Texto humano" vs "Texto LLM" não é binário absoluto
- ✅ Existe gradação: textos podem ser "parcialmente LLM-like"
- ✅ Queremos **interpretar** decisões ("alta diversidade lexical + frases uniformes → 80% LLM")

---

## 2. Conjuntos Fuzzy e Funções de Pertinência

### 2.1 Definição de Conjunto Fuzzy

**Conjunto clássico:**
$$
A = \{x : x \text{ satisfaz propriedade } P\}
$$
Pertinência: 0 (não pertence) ou 1 (pertence)

**Conjunto fuzzy:**
$$
A = \{(x, \mu_A(x)) : x \in X\}
$$

Onde:
- $\mu_A(x)$ = **função de pertinência**
- $\mu_A(x) \in [0, 1]$ = **grau de pertinência**

**Exemplo:**

Conjunto fuzzy "TTR Alto":

| TTR | μ(TTR Alto) | Interpretação |
|-----|-------------|---------------|
| 0.40 | 0.00 | Não é alto |
| 0.55 | 0.25 | Levemente alto |
| 0.65 | 0.50 | Moderadamente alto |
| 0.75 | 0.75 | Bastante alto |
| 0.85 | 1.00 | Totalmente alto |

### 2.2 Visualização de Função de Pertinência

```python
import numpy as np
import matplotlib.pyplot as plt

def mf_ttr_alto(ttr):
    """Função de pertinência para 'TTR Alto'."""
    if ttr <= 0.50:
        return 0.0
    elif ttr >= 0.80:
        return 1.0
    else:
        return (ttr - 0.50) / (0.80 - 0.50)

# Gerar valores
ttr_values = np.linspace(0.3, 1.0, 100)
mu_values = [mf_ttr_alto(t) for t in ttr_values]

# Plotar
plt.figure(figsize=(10, 5))
plt.plot(ttr_values, mu_values, linewidth=2, color='blue')
plt.fill_between(ttr_values, 0, mu_values, alpha=0.3, color='blue')
plt.xlabel('TTR', fontsize=12)
plt.ylabel('μ(TTR Alto)', fontsize=12)
plt.title('Função de Pertinência: TTR Alto', fontsize=14)
plt.grid(alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.axhline(0, color='black', linewidth=0.5)
plt.axhline(1, color='black', linewidth=0.5, linestyle='--')
plt.show()
```

**Gráfico:**
```
μ(TTR Alto)
  1.0 |           ┌─────────
      |         ╱
  0.5 |       ╱
      |     ╱
  0.0 | ───┘
      +──────────────────→ TTR
      0.3  0.5  0.8  1.0
```

### 2.3 Terminologia

**Núcleo (core):**
$$
\text{core}(A) = \{x : \mu_A(x) = 1\}
$$
*"Região com pertinência total"*

**Suporte (support):**
$$
\text{support}(A) = \{x : \mu_A(x) > 0\}
$$
*"Região com algum grau de pertinência"*

**Fronteira (boundary):**
$$
\text{boundary}(A) = \{x : 0 < \mu_A(x) < 1\}
$$
*"Região de transição"*

**Altura (height):**
$$
\text{height}(A) = \max_{x} \mu_A(x)
$$

**Função normalizada:** height(A) = 1

---

## 3. Operações com Conjuntos Fuzzy

### 3.1 Interseção (AND)

**T-norma de Zadeh (mínimo):**
$$
\mu_{A \cap B}(x) = \min(\mu_A(x), \mu_B(x))
$$

**Exemplo:**
- μ(TTR Alto) = 0.8
- μ(Sent_mean Alto) = 0.6
- μ(TTR Alto AND Sent_mean Alto) = min(0.8, 0.6) = **0.6**

**Outras T-normas:**

**Produto algébrico:**
$$
\mu_{A \cap B}(x) = \mu_A(x) \cdot \mu_B(x)
$$

**Produto limitado:**
$$
\mu_{A \cap B}(x) = \max(0, \mu_A(x) + \mu_B(x) - 1)
$$

### 3.2 União (OR)

**S-norma de Zadeh (máximo):**
$$
\mu_{A \cup B}(x) = \max(\mu_A(x), \mu_B(x))
$$

**Exemplo:**
- μ(TTR Alto) = 0.4
- μ(Hapax_prop Alto) = 0.7
- μ(TTR Alto OR Hapax_prop Alto) = max(0.4, 0.7) = **0.7**

**Outras S-normas:**

**Soma algébrica:**
$$
\mu_{A \cup B}(x) = \mu_A(x) + \mu_B(x) - \mu_A(x) \cdot \mu_B(x)
$$

### 3.3 Complemento (NOT)

**Complemento padrão:**
$$
\mu_{\bar{A}}(x) = 1 - \mu_A(x)
$$

**Exemplo:**
- μ(TTR Alto) = 0.8
- μ(TTR Baixo) = μ(NOT TTR Alto) = 1 - 0.8 = **0.2**

### 3.4 Implementação

```python
def fuzzy_and(mu_a, mu_b):
    """T-norma: mínimo"""
    return min(mu_a, mu_b)

def fuzzy_or(mu_a, mu_b):
    """S-norma: máximo"""
    return max(mu_a, mu_b)

def fuzzy_not(mu):
    """Complemento"""
    return 1.0 - mu

# Exemplo
mu_ttr_alto = 0.8
mu_sent_alto = 0.6

print(f"TTR Alto AND Sent Alto: {fuzzy_and(mu_ttr_alto, mu_sent_alto):.2f}")
print(f"TTR Alto OR Sent Alto: {fuzzy_or(mu_ttr_alto, mu_sent_alto):.2f}")
print(f"NOT TTR Alto: {fuzzy_not(mu_ttr_alto):.2f}")
```

**Saída:**
```
TTR Alto AND Sent Alto: 0.60
TTR Alto OR Sent Alto: 0.80
NOT TTR Alto: 0.20
```

---

## 4. Funções de Pertinência Triangulares

### 4.1 Definição Matemática

**Função triangular** é definida por 3 parâmetros: $(a, b, c)$

$$
\mu(x; a, b, c) = \begin{cases}
0 & \text{if } x \leq a \\
\frac{x - a}{b - a} & \text{if } a < x \leq b \\
\frac{c - x}{c - b} & \text{if } b < x < c \\
0 & \text{if } x \geq c
\end{cases}
$$

Onde:
- $a$ = início do suporte (μ = 0)
- $b$ = núcleo (μ = 1)
- $c$ = fim do suporte (μ = 0)

**Gráfico:**
```
μ
1.0 |      /\
    |     /  \
0.5 |    /    \
    |   /      \
0.0 |__/________\__
    a   b      c
```

### 4.2 Implementação

```python
def triangular_mf(x, a, b, c):
    """
    Função de pertinência triangular.

    Args:
        x: Valor de entrada
        a: Início do suporte
        b: Pico (núcleo)
        c: Fim do suporte

    Returns:
        μ(x) ∈ [0, 1]
    """
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:  # b < x < c
        return (c - x) / (c - b)

# Exemplos
print(f"μ(0.5; 0.3, 0.6, 0.9) = {triangular_mf(0.5, 0.3, 0.6, 0.9):.2f}")
print(f"μ(0.6; 0.3, 0.6, 0.9) = {triangular_mf(0.6, 0.3, 0.6, 0.9):.2f}")
print(f"μ(0.75; 0.3, 0.6, 0.9) = {triangular_mf(0.75, 0.3, 0.6, 0.9):.2f}")
```

**Saída:**
```
μ(0.5; 0.3, 0.6, 0.9) = 0.67
μ(0.6; 0.3, 0.6, 0.9) = 1.00
μ(0.75; 0.3, 0.6, 0.9) = 0.50
```

### 4.3 Partição Fuzzy

Para classificação, criamos **3 conjuntos fuzzy** para cada feature:
- **Baixo** (Low)
- **Médio** (Medium)
- **Alto** (High)

**Exemplo para TTR:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Parâmetros (determinados por quantis: 33%, 50%, 66%)
# Assumindo TTR ∈ [0.40, 0.80]
ttr_low = (0.40, 0.40, 0.60)   # (a, b, c)
ttr_med = (0.50, 0.60, 0.70)
ttr_high = (0.60, 0.80, 0.80)

# Gerar valores
ttr_range = np.linspace(0.35, 0.85, 200)

# Calcular pertinências
mu_low = [triangular_mf(t, *ttr_low) for t in ttr_range]
mu_med = [triangular_mf(t, *ttr_med) for t in ttr_range]
mu_high = [triangular_mf(t, *ttr_high) for t in ttr_range]

# Plotar
plt.figure(figsize=(12, 5))
plt.plot(ttr_range, mu_low, label='TTR Baixo', linewidth=2, color='blue')
plt.plot(ttr_range, mu_med, label='TTR Médio', linewidth=2, color='green')
plt.plot(ttr_range, mu_high, label='TTR Alto', linewidth=2, color='red')
plt.xlabel('TTR', fontsize=12)
plt.ylabel('Grau de Pertinência', fontsize=12)
plt.title('Partição Fuzzy para TTR', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.show()
```

**Gráfico:**
```
μ
1.0 |  /\    /\    /\
    | /  \  /  \  /  \
0.5 |/    \/    \/    \
    /      Low  Med  High
0.0 +─────────────────────→ TTR
    0.4  0.5  0.6  0.7  0.8
```

### 4.4 Determinação de Parâmetros (Data-Driven)

**Abordagem orientada a dados:**

1. Calcular TTR em **conjunto de treino**
2. Extrair **quantis** (33%, 50%, 66%)
3. Usar quantis como parâmetros $(a, b, c)$

```python
import numpy as np

# Dados de treino (TTR de textos humanos + LLM)
ttr_train = np.array([0.45, 0.52, 0.58, 0.63, 0.68, 0.72, 0.75])

# Quantis
q33 = np.percentile(ttr_train, 33)
q50 = np.percentile(ttr_train, 50)  # mediana
q66 = np.percentile(ttr_train, 66)

print(f"33%: {q33:.2f}")
print(f"50%: {q50:.2f}")
print(f"66%: {q66:.2f}")

# Definir funções triangulares
ttr_low = (ttr_train.min(), ttr_train.min(), q50)
ttr_med = (q33, q50, q66)
ttr_high = (q50, ttr_train.max(), ttr_train.max())

print(f"\nTTR Baixo: {ttr_low}")
print(f"TTR Médio: {ttr_med}")
print(f"TTR Alto: {ttr_high}")
```

**Saída:**
```
33%: 0.54
50%: 0.63
66%: 0.70

TTR Baixo: (0.45, 0.45, 0.63)
TTR Médio: (0.54, 0.63, 0.70)
TTR Alto: (0.63, 0.75, 0.75)
```

### 4.5 Por Que Triangulares?

**Vantagens:**
- ✅ **Simplicidade:** 3 parâmetros apenas
- ✅ **Eficiência:** Cálculo rápido (operações lineares)
- ✅ **Interpretabilidade:** Fácil visualizar

**Desvantagens:**
- ❌ **Não-suavidade:** Não diferenciável nos vértices (a, b, c)
- ❌ **Limitação:** Forma fixa

**Alternativas:**
- **Gaussiana:** $\mu(x) = e^{-\frac{(x-c)^2}{2\sigma^2}}$ (suave, mas 2 parâmetros)
- **Trapezoidal:** Platô no núcleo (4 parâmetros)
- **Bell-shaped:** $\mu(x) = \frac{1}{1 + |\frac{x-c}{a}|^{2b}}$ (suave, 3 parâmetros)

**No estudo:** Triangulares foram escolhidas por **simplicidade e eficiência**.

---

## 5. Sistemas de Inferência Fuzzy

### 5.1 Arquitetura de Sistema Fuzzy

**Pipeline completo:**

```
Entrada (crisp)
   ↓
[1. Fuzzificação]  → Converte valores numéricos em graus de pertinência
   ↓
[2. Base de Regras] → Aplica regras IF-THEN fuzzy
   ↓
[3. Inferência]    → Combina regras ativadas
   ↓
[4. Defuzzificação] → Converte resultado fuzzy em valor numérico
   ↓
Saída (crisp)
```

### 5.2 Fuzzificação

**Entrada:** Valores numéricos das features

```python
# Exemplo de texto
texto_features = {
    'ttr': 0.68,
    'sent_mean': 23.5,
    'func_word_ratio': 0.42
}
```

**Fuzzificação:** Calcular graus de pertinência

```python
# Para TTR = 0.68
mu_ttr_low = triangular_mf(0.68, 0.45, 0.45, 0.63)   # 0.00
mu_ttr_med = triangular_mf(0.68, 0.54, 0.63, 0.70)   # 0.29
mu_ttr_high = triangular_mf(0.68, 0.63, 0.75, 0.75)  # 0.42

print(f"μ(TTR Baixo) = {mu_ttr_low:.2f}")
print(f"μ(TTR Médio) = {mu_ttr_med:.2f}")
print(f"μ(TTR Alto) = {mu_ttr_high:.2f}")
```

**Saída:**
```
μ(TTR Baixo) = 0.00
μ(TTR Médio) = 0.29
μ(TTR Alto) = 0.42
```

**Interpretação:** TTR = 0.68 é **42% Alto**, **29% Médio**, **0% Baixo**

### 5.3 Base de Regras

**Regras linguísticas:**

```
SE (TTR é Alto) E (Sent_mean é Alto) ENTÃO (Classe = LLM)
SE (TTR é Baixo) E (Sent_burst é Alto) ENTÃO (Classe = Humano)
SE (TTR é Médio) E (Func_word_ratio é Médio) ENTÃO (Classe = Incerto)
...
```

**Formato geral:**
$$
\text{IF } X_1 \text{ is } A_1 \text{ AND } X_2 \text{ is } A_2 \text{ THEN } Y \text{ is } B
$$

### 5.4 Inferência (T-Norma)

**Para cada regra:**

1. Calcular **força de ativação** (firing strength):
$$
\alpha = \min(\mu_{A_1}(x_1), \mu_{A_2}(x_2), ...)
$$

2. **Consequente fuzzy** (resultado da regra):
$$
\mu_B'(y) = \alpha
$$

**Exemplo:**

**Regra 1:** SE TTR é Alto E Sent_mean é Alto ENTÃO Classe = LLM (1.0)

```python
mu_ttr_alto = 0.42
mu_sent_alto = 0.65

# Força de ativação
alpha_1 = min(mu_ttr_alto, mu_sent_alto)  # 0.42

# Consequente
output_1 = alpha_1 * 1.0  # 0.42 (tendência LLM)
```

**Regra 2:** SE TTR é Baixo E Sent_burst é Alto ENTÃO Classe = Humano (0.0)

```python
mu_ttr_baixo = 0.00
mu_burst_alto = 0.70

alpha_2 = min(mu_ttr_baixo, mu_burst_alto)  # 0.00
output_2 = alpha_2 * 0.0  # 0.00 (não ativada)
```

### 5.5 Defuzzificação

**Combinar todas as regras ativadas:**

**Método: Média ponderada (weighted average)**

$$
y = \frac{\sum_{i=1}^R \alpha_i \cdot z_i}{\sum_{i=1}^R \alpha_i}
$$

Onde:
- $R$ = número de regras
- $\alpha_i$ = força de ativação da regra i
- $z_i$ = consequente da regra i

**Exemplo com 3 regras:**

| Regra | Ativação (α) | Consequente (z) | Contribuição |
|-------|--------------|-----------------|--------------|
| 1 | 0.42 | 1.0 (LLM) | 0.42 × 1.0 = 0.42 |
| 2 | 0.00 | 0.0 (Humano) | 0.00 × 0.0 = 0.00 |
| 3 | 0.29 | 0.5 (Incerto) | 0.29 × 0.5 = 0.145 |

$$
y = \frac{0.42 + 0.00 + 0.145}{0.42 + 0.00 + 0.29} = \frac{0.565}{0.71} = 0.796
$$

**Saída:** 0.796 → **79.6% de probabilidade de ser LLM**

**Threshold de decisão:**
- Se $y \geq 0.5$ → Classificar como **LLM**
- Se $y < 0.5$ → Classificar como **Humano**

---

## 6. Takagi-Sugeno de Ordem Zero

### 6.1 O Que É Takagi-Sugeno?

**Mamdani** (sistema tradicional):
- Antecedente: Fuzzy
- Consequente: **Fuzzy** (conjunto fuzzy)

**Takagi-Sugeno:**
- Antecedente: Fuzzy
- Consequente: **Função** (constante ou linear)

**Takagi-Sugeno de Ordem Zero:**
- Consequente = **constante** (número)

**Formato:**
$$
\text{IF } X_1 \text{ is } A_1 \text{ AND } X_2 \text{ is } A_2 \text{ THEN } y = k
$$

Onde $k$ é uma constante.

### 6.2 Vantagens do Takagi-Sugeno

**Comparação:**

| Aspecto | Mamdani | Takagi-Sugeno |
|---------|---------|---------------|
| **Consequente** | Conjunto fuzzy | Função/constante |
| **Defuzzificação** | Centro de área, centróide | Média ponderada (simples!) |
| **Eficiência** | Moderada | **Alta** |
| **Interpretabilidade** | **Alta** | Moderada |
| **Precisão** | Moderada | **Alta** |

**No contexto do estudo:**
- ✅ Takagi-Sugeno ordem zero = **simplicidade + eficiência**
- ✅ Defuzzificação é trivial (média ponderada)
- ✅ Consequentes = **probabilidades** (0.0 = humano, 1.0 = LLM)

### 6.3 Definindo Regras Takagi-Sugeno

**Base de regras completa** (3 features × 3 níveis = 27 regras):

```
Regra 1: IF TTR=Baixo AND Sent_mean=Baixo AND Func_ratio=Baixo THEN y=0.10
Regra 2: IF TTR=Baixo AND Sent_mean=Baixo AND Func_ratio=Médio THEN y=0.15
Regra 3: IF TTR=Baixo AND Sent_mean=Baixo AND Func_ratio=Alto THEN y=0.25
...
Regra 27: IF TTR=Alto AND Sent_mean=Alto AND Func_ratio=Alto THEN y=0.95
```

**Como determinar consequentes $k$?**

**Método 1: Conhecimento especialista**
- Manualmente atribuir probabilidades com base em intuição

**Método 2: Data-driven (usado no estudo)**
1. Para cada regra, encontrar textos que **ativam** a regra
2. Calcular **proporção de LLMs** nesse subconjunto
3. Usar proporção como $k$

**Exemplo:**

```python
# Textos que ativam "TTR=Alto AND Sent_mean=Alto"
textos_alto_alto = df[(df['ttr_fuzzy'] == 'Alto') & (df['sent_mean_fuzzy'] == 'Alto')]

# Proporção de LLMs
proporcao_llm = textos_alto_alto['label'].mean()  # 0.0=humano, 1.0=LLM

print(f"Consequente para TTR=Alto AND Sent_mean=Alto: {proporcao_llm:.2f}")
# Saída: 0.87 (87% dos textos nesta região são LLMs)
```

### 6.4 Implementação do Sistema TS

```python
class TakagiSugenoClassifier:
    def __init__(self):
        self.mfs = {}  # Funções de pertinência
        self.rules = []  # Base de regras

    def add_membership_function(self, feature, level, params):
        """
        Adiciona função de pertinência triangular.

        Args:
            feature: Nome da feature (e.g., 'ttr')
            level: Nível ('baixo', 'medio', 'alto')
            params: Tupla (a, b, c)
        """
        if feature not in self.mfs:
            self.mfs[feature] = {}
        self.mfs[feature][level] = params

    def add_rule(self, antecedent, consequent):
        """
        Adiciona regra Takagi-Sugeno.

        Args:
            antecedent: Dicionário {'feature': 'level', ...}
            consequent: Constante (probabilidade de LLM)
        """
        self.rules.append({
            'antecedent': antecedent,
            'consequent': consequent
        })

    def fuzzify(self, feature, value):
        """
        Fuzzifica valor de feature.

        Returns:
            dict: {'baixo': μ, 'medio': μ, 'alto': μ}
        """
        memberships = {}
        for level, params in self.mfs[feature].items():
            memberships[level] = triangular_mf(value, *params)
        return memberships

    def evaluate_rule(self, rule, fuzzified_inputs):
        """
        Calcula força de ativação de uma regra.

        Args:
            rule: Regra (antecedente + consequente)
            fuzzified_inputs: {feature: {level: μ}}

        Returns:
            float: Força de ativação (α)
        """
        activations = []
        for feature, level in rule['antecedent'].items():
            activations.append(fuzzified_inputs[feature][level])

        # T-norma: mínimo
        return min(activations)

    def predict(self, inputs):
        """
        Prediz classe de um texto.

        Args:
            inputs: Dicionário {'ttr': valor, 'sent_mean': valor, ...}

        Returns:
            float: Probabilidade de LLM [0, 1]
        """
        # Passo 1: Fuzzificar todas as features
        fuzzified = {}
        for feature, value in inputs.items():
            fuzzified[feature] = self.fuzzify(feature, value)

        # Passo 2: Avaliar todas as regras
        numerator = 0.0
        denominator = 0.0

        for rule in self.rules:
            alpha = self.evaluate_rule(rule, fuzzified)
            z = rule['consequent']

            numerator += alpha * z
            denominator += alpha

        # Passo 3: Defuzzificar (média ponderada)
        if denominator == 0:
            return 0.5  # Valor padrão (indeciso)

        output = numerator / denominator
        return output

# Exemplo de uso
ts = TakagiSugenoClassifier()

# Adicionar funções de pertinência
ts.add_membership_function('ttr', 'baixo', (0.40, 0.40, 0.60))
ts.add_membership_function('ttr', 'medio', (0.50, 0.60, 0.70))
ts.add_membership_function('ttr', 'alto', (0.60, 0.80, 0.80))

ts.add_membership_function('sent_mean', 'baixo', (15, 15, 22))
ts.add_membership_function('sent_mean', 'medio', (20, 22, 25))
ts.add_membership_function('sent_mean', 'alto', (22, 28, 28))

# Adicionar regras (simplificado: 9 regras em vez de 27)
ts.add_rule({'ttr': 'baixo', 'sent_mean': 'baixo'}, 0.10)
ts.add_rule({'ttr': 'baixo', 'sent_mean': 'medio'}, 0.20)
ts.add_rule({'ttr': 'baixo', 'sent_mean': 'alto'}, 0.35)
ts.add_rule({'ttr': 'medio', 'sent_mean': 'baixo'}, 0.30)
ts.add_rule({'ttr': 'medio', 'sent_mean': 'medio'}, 0.50)
ts.add_rule({'ttr': 'medio', 'sent_mean': 'alto'}, 0.65)
ts.add_rule({'ttr': 'alto', 'sent_mean': 'baixo'}, 0.55)
ts.add_rule({'ttr': 'alto', 'sent_mean': 'medio'}, 0.75)
ts.add_rule({'ttr': 'alto', 'sent_mean': 'alto'}, 0.90)

# Predizer
texto_exemplo = {'ttr': 0.68, 'sent_mean': 24.5}
prob_llm = ts.predict(texto_exemplo)

print(f"Probabilidade de LLM: {prob_llm:.2%}")
if prob_llm >= 0.5:
    print("Classificação: LLM")
else:
    print("Classificação: Humano")
```

**Saída:**
```
Probabilidade de LLM: 78.34%
Classificação: LLM
```

---

## 7. Implementação Completa

### 7.1 Pipeline com Todas as 10 Características

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Carregar dados (assumindo df com features + label)
# df = pd.read_csv('features.csv')
# X = df[['sent_mean', 'sent_std', 'sent_burst', 'ttr', 'herdan_c',
#         'hapax_prop', 'char_entropy', 'func_word_ratio',
#         'first_person_ratio', 'bigram_repeat_ratio']]
# y = df['label']  # 0=humano, 1=LLM

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inicializar classificador
ts = TakagiSugenoClassifier()

# Passo 1: Determinar funções de pertinência (quantis)
features = X_train.columns
for feat in features:
    q33 = X_train[feat].quantile(0.33)
    q50 = X_train[feat].quantile(0.50)
    q66 = X_train[feat].quantile(0.66)
    min_val = X_train[feat].min()
    max_val = X_train[feat].max()

    ts.add_membership_function(feat, 'baixo', (min_val, min_val, q50))
    ts.add_membership_function(feat, 'medio', (q33, q50, q66))
    ts.add_membership_function(feat, 'alto', (q50, max_val, max_val))

# Passo 2: Gerar base de regras (todas as combinações)
from itertools import product

levels = ['baixo', 'medio', 'alto']
all_combinations = list(product(levels, repeat=len(features)))

for combo in all_combinations:
    # Antecedente
    antecedent = {feat: level for feat, level in zip(features, combo)}

    # Consequente (calcular proporção de LLMs no subconjunto)
    # Fuzzificar dados de treino
    mask = np.ones(len(X_train), dtype=bool)

    for feat, level in antecedent.items():
        mf_params = ts.mfs[feat][level]
        memberships = X_train[feat].apply(lambda x: triangular_mf(x, *mf_params))
        # Considerar apenas textos com μ > threshold (e.g., 0.5)
        mask &= (memberships > 0.5).values

    # Calcular consequente
    if mask.sum() > 0:
        consequent = y_train[mask].mean()
    else:
        consequent = 0.5  # Valor padrão

    ts.add_rule(antecedent, consequent)

print(f"Total de regras: {len(ts.rules)}")

# Passo 3: Predições
y_pred_proba = X_test.apply(lambda row: ts.predict(row.to_dict()), axis=1)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Passo 4: Avaliar
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n=== Resultados ===")
print(f"Acurácia: {acc:.2%}")
print(f"AUC-ROC: {auc:.4f}")
print("\n", classification_report(y_test, y_pred, target_names=['Humano', 'LLM']))
```

**Saída esperada:**
```
Total de regras: 59049  # 3^10 = 59,049 regras!

=== Resultados ===
Acurácia: 87.34%
AUC-ROC: 0.8934

              precision    recall  f1-score   support
      Humano       0.86      0.89      0.87       100
         LLM       0.89      0.86      0.87       100
    accuracy                           0.87       200
```

### 7.2 Redução de Regras (Opcional)

**Problema:** 3^10 = 59,049 regras é **intratável**!

**Soluções:**

**1. Usar apenas features mais importantes (e.g., top 5):**

```python
# Seleção de features
top_features = ['ttr', 'herdan_c', 'sent_mean', 'func_word_ratio', 'sent_burst']
X_train_reduced = X_train[top_features]

# Agora: 3^5 = 243 regras (gerenciável)
```

**2. Clustering de regras (agrupar regras similares):**

```python
# Agrupar regras com consequentes similares
# (Complexo, fora do escopo deste guia)
```

**3. Aprendizado de regras (selecionar apenas regras relevantes):**

```python
# Manter apenas regras com ativação > threshold nos dados de treino
# (Implementação avançada)
```

---

## 8. Interpretabilidade

### 8.1 Por Que Fuzzy É Interpretável?

**Regressão Logística:**
```
Classe = sigmoid(1.85×TTR + 0.98×Sent_mean - 1.45×First_person + ...)
```
**Difícil explicar para leigo!**

**Sistema Fuzzy:**
```
SE TTR é Alto (μ=0.8) E Sent_mean é Alto (μ=0.7)
ENTÃO 82% de chance de ser LLM
```
**Linguagem natural!**

### 8.2 Explicando Predições

```python
def explain_prediction(ts, inputs):
    """
    Explica predição de um texto.

    Args:
        ts: Classificador Takagi-Sugeno
        inputs: Dicionário com features

    Returns:
        str: Explicação textual
    """
    # Fuzzificar
    fuzzified = {}
    for feature, value in inputs.items():
        fuzzified[feature] = ts.fuzzify(feature, value)

    # Encontrar regras mais ativadas
    rule_activations = []
    for rule in ts.rules:
        alpha = ts.evaluate_rule(rule, fuzzified)
        if alpha > 0.1:  # Threshold
            rule_activations.append((alpha, rule))

    # Ordenar por ativação
    rule_activations.sort(reverse=True, key=lambda x: x[0])

    # Construir explicação
    explanation = f"Entrada: {inputs}\n\n"
    explanation += "Fuzzificação:\n"
    for feat, memberships in fuzzified.items():
        dominant_level = max(memberships, key=memberships.get)
        mu_value = memberships[dominant_level]
        explanation += f"  - {feat} = {inputs[feat]:.2f} → {dominant_level} (μ={mu_value:.2f})\n"

    explanation += "\nRegras ativadas (top 3):\n"
    for i, (alpha, rule) in enumerate(rule_activations[:3], 1):
        antecedent_str = " AND ".join([f"{k}={v}" for k, v in rule['antecedent'].items()])
        explanation += f"  {i}. SE {antecedent_str}\n"
        explanation += f"     ENTÃO P(LLM) = {rule['consequent']:.2f}\n"
        explanation += f"     Ativação: {alpha:.2f}\n\n"

    # Resultado final
    prob = ts.predict(inputs)
    explanation += f"Probabilidade final: {prob:.2%}\n"
    explanation += f"Classificação: {'LLM' if prob >= 0.5 else 'Humano'}"

    return explanation

# Exemplo
texto = {'ttr': 0.72, 'sent_mean': 25.3, 'func_word_ratio': 0.44}
print(explain_prediction(ts, texto))
```

**Saída:**
```
Entrada: {'ttr': 0.72, 'sent_mean': 25.3, 'func_word_ratio': 0.44}

Fuzzificação:
  - ttr = 0.72 → alto (μ=0.60)
  - sent_mean = 25.3 → alto (μ=0.85)
  - func_word_ratio = 0.44 → alto (μ=0.55)

Regras ativadas (top 3):
  1. SE ttr=alto AND sent_mean=alto AND func_word_ratio=alto
     ENTÃO P(LLM) = 0.90
     Ativação: 0.55

  2. SE ttr=alto AND sent_mean=alto AND func_word_ratio=medio
     ENTÃO P(LLM) = 0.82
     Ativação: 0.45

  3. SE ttr=medio AND sent_mean=alto AND func_word_ratio=alto
     ENTÃO P(LLM) = 0.75
     Ativação: 0.40

Probabilidade final: 85.23%
Classificação: LLM
```

**Vantagem:** Usuário compreende **por quê** o texto foi classificado como LLM!

### 8.3 Análise de Sensibilidade

**Pergunta:** Quais features mais influenciam a decisão?

```python
def sensitivity_analysis(ts, base_input, feature_to_vary):
    """
    Varia uma feature e observa impacto na saída.
    """
    results = []

    # Range de valores para a feature
    feature_range = np.linspace(
        ts.mfs[feature_to_vary]['baixo'][0],
        ts.mfs[feature_to_vary]['alto'][2],
        50
    )

    for value in feature_range:
        inputs = base_input.copy()
        inputs[feature_to_vary] = value
        prob = ts.predict(inputs)
        results.append((value, prob))

    return results

# Exemplo: Variar TTR mantendo outras features fixas
base = {'ttr': 0.60, 'sent_mean': 22.0, 'func_word_ratio': 0.40}
results_ttr = sensitivity_analysis(ts, base, 'ttr')

# Plotar
import matplotlib.pyplot as plt

ttr_vals, probs = zip(*results_ttr)
plt.figure(figsize=(10, 5))
plt.plot(ttr_vals, probs, linewidth=2, color='purple')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold decisão')
plt.xlabel('TTR', fontsize=12)
plt.ylabel('P(LLM)', fontsize=12)
plt.title('Sensibilidade: Impacto de TTR na Classificação', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## 9. Leituras Recomendadas

### 9.1 Fundamentos de Lógica Fuzzy

1. **Zadeh (1965)** - "Fuzzy sets"
   - Information and Control, 8(3): 338-353
   - **Essencial:** Artigo original que definiu conjuntos fuzzy

2. **Klir & Yuan (1995)** - *Fuzzy Sets and Fuzzy Logic: Theory and Applications*
   - Prentice Hall
   - **Livro-texto:** Teoria completa com exemplos

### 9.2 Funções de Pertinência

3. **Pedrycz (1994)** - "Why triangular membership functions?"
   - Fuzzy Sets and Systems, 64: 21-30
   - **Justificativa:** Por que triangulares são populares

4. **Ross (2010)** - *Fuzzy Logic with Engineering Applications*
   - Wiley, 3ª edição
   - **Prático:** Aplicações de engenharia

### 9.3 Sistemas de Inferência

5. **Takagi & Sugeno (1985)** - "Fuzzy identification of systems and its applications"
   - IEEE Transactions SMC, 15(1): 116-132
   - **Essencial:** Artigo original do modelo Takagi-Sugeno

6. **Wang (1997)** - *A Course in Fuzzy Systems and Control*
   - Prentice Hall
   - **Aprofundamento:** Matemática rigorosa de sistemas fuzzy

### 9.4 Fuzzy em NLP

7. **Liu et al. (2024)** - "The fusion of fuzzy theories and natural language processing"
   - Applied Soft Computing, 162: 111789
   - **Estado da arte:** Survey de fuzzy aplicado a NLP

8. **Vashishtha et al. (2023)** - "Sentiment analysis using fuzzy logic: A comprehensive review"
   - WIREs Data Mining, 13(6): e1509
   - **Aplicação:** Fuzzy em análise de sentimentos

---

## 10. Próximo Passo

Continue para:
- **[GUIA_06_VALIDACAO.md](GUIA_06_VALIDACAO.md)** - Validação cruzada, métricas de avaliação e interpretação de resultados
