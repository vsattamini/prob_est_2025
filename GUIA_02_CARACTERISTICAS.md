# Guia Completo do Projeto - Parte 2: Características Estilométricas Detalhadas

**Pré-requisito:** Leia [GUIA_01_VISAO_GERAL.md](GUIA_01_VISAO_GERAL.md) primeiro

---

## 1. Implementação Passo a Passo: Como Extrair Cada Característica

### 1.1 Preparação do Texto

Antes de extrair qualquer característica, o texto passa por pré-processamento básico:

```python
import re
import unicodedata

def preprocessar_texto(text):
    """
    Normaliza o texto sem perder informação importante.
    """
    # 1. Normalização Unicode (NFD → NFC)
    text = unicodedata.normalize('NFC', text)

    # 2. Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Exemplo:
texto_bruto = "O  gato    viu\no rato."
texto_limpo = preprocessar_texto(texto_bruto)
# Resultado: "O gato viu o rato."
```

**⚠️ IMPORTANTE:** NÃO removemos pontuação, pois ela é importante para:
- Segmentação de frases
- Cálculo de entropia de caracteres
- Análise estilométrica geral

---

### 1.2 Segmentação de Frases

**Bibliotecas usadas:** NLTK (Natural Language Toolkit)

```python
import nltk
# Baixar modelo de segmentação para português (fazer uma vez)
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def segmentar_frases(text, lang='portuguese'):
    """
    Divide o texto em frases usando o tokenizador NLTK.
    """
    sentences = sent_tokenize(text, language=lang)
    return sentences

# Exemplo:
texto = "O gato está dormindo. Ele gosta de leite. Por quê? Não sei!"
frases = segmentar_frases(texto)
# Resultado: ['O gato está dormindo.', 'Ele gosta de leite.', 'Por quê?', 'Não sei!']
```

**Como funciona internamente:**
- NLTK usa modelos treinados (Punkt Sentence Tokenizer)
- Reconhece abreviações ("Sr.", "Dr.", "etc.")
- Distingue pontos de final de frase vs pontos em abreviações
- Lida com pontuação portuguesa (?, !, ..., —)

---

### 1.3 Tokenização de Palavras

**Método simples:** Regex para dividir em tokens

```python
import re

def tokenizar_palavras(text):
    """
    Divide o texto em palavras (tokens).
    Mantém apenas palavras alfanuméricas, descarta pontuação.
    """
    # Padrão: sequências de letras/números (incluindo acentos)
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

# Exemplo:
texto = "O gato está dormindo! Ele gosta de leite?"
tokens = tokenizar_palavras(texto)
# Resultado: ['o', 'gato', 'está', 'dormindo', 'ele', 'gosta', 'de', 'leite']
```

**Alternativa mais sofisticada com NLTK:**

```python
from nltk.tokenize import word_tokenize

def tokenizar_palavras_nltk(text, lang='portuguese'):
    """
    Tokenização mais precisa usando NLTK.
    """
    tokens = word_tokenize(text.lower(), language=lang)
    # Filtrar apenas palavras (remover pontuação)
    tokens = [t for t in tokens if t.isalnum()]
    return tokens
```

---

## 2. Características de Frase (sent_mean, sent_std, sent_burst)

### 2.1 sent_mean - Comprimento Médio de Frase

**Código completo:**

```python
def calcular_sent_mean(text):
    """
    Calcula o comprimento médio das frases em palavras.

    Retorna:
        float: Número médio de palavras por frase
    """
    # Segmentar em frases
    sentences = segmentar_frases(text)

    if not sentences:
        return 0.0

    # Contar palavras em cada frase
    sentence_lengths = []
    for sentence in sentences:
        words = tokenizar_palavras(sentence)
        sentence_lengths.append(len(words))

    # Calcular média
    sent_mean = sum(sentence_lengths) / len(sentence_lengths)
    return sent_mean

# Exemplo detalhado:
texto = """
O gato dormia tranquilamente sob a árvore.
Ele sonhava com peixes.
Por que os gatos adoram peixes?
Ninguém sabe ao certo.
"""

frases = segmentar_frases(texto)
# Frase 1: 7 palavras
# Frase 2: 4 palavras
# Frase 3: 6 palavras
# Frase 4: 4 palavras
# Média: (7+4+6+4)/4 = 5.25

resultado = calcular_sent_mean(texto)
print(f"Comprimento médio: {resultado:.2f} palavras/frase")
# Output: Comprimento médio: 5.25 palavras/frase
```

**Interpretação:**
- **sent_mean baixo** (3-5 palavras) → Estilo telegráfico, frases curtas
- **sent_mean médio** (8-15 palavras) → Estilo normal, conversacional
- **sent_mean alto** (20+ palavras) → Estilo complexo, acadêmico

**Diferença humano vs LLM:**
- Δ = +0.126 (negligenciável) → Praticamente igual
- Humanos: média ~11.5 palavras
- LLMs: média ~11.8 palavras

---

### 2.2 sent_std - Desvio Padrão do Comprimento de Frase

**Código completo:**

```python
import math

def calcular_sent_std(text):
    """
    Calcula o desvio padrão dos comprimentos de frase.

    Retorna:
        float: Desvio padrão em número de palavras
    """
    sentences = segmentar_frases(text)

    if len(sentences) < 2:
        return 0.0

    # Comprimentos
    lengths = [len(tokenizar_palavras(s)) for s in sentences]

    # Média
    mean = sum(lengths) / len(lengths)

    # Variância: média dos quadrados das diferenças
    variance = sum((x - mean)**2 for x in lengths) / (len(lengths) - 1)

    # Desvio padrão: raiz quadrada da variância
    std = math.sqrt(variance)

    return std

# Exemplo:
# Texto com frases uniformes
texto_uniforme = """
O gato dorme.
O rato corre.
O cão late.
"""
std_uniforme = calcular_sent_std(texto_uniforme)
print(f"Std (uniforme): {std_uniforme:.2f}")  # ~0.0 (todas iguais)

# Texto com frases variadas
texto_variado = """
O gato dorme.
Ele sonha com ratos que correm pela casa toda.
Por quê?
"""
std_variado = calcular_sent_std(texto_variado)
print(f"Std (variado): {std_variado:.2f}")  # ~3.5 (alta variação)
```

**Fórmula matemática:**

$$
\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

Onde:
- n = número de frases
- $x_i$ = comprimento da frase i
- $\bar{x}$ = comprimento médio

**Por que dividimos por (n-1) e não n?**
- Usamos **desvio padrão amostral** (sample std)
- Correção de Bessel: ajusta viés em amostras pequenas
- Se tivéssemos a população inteira, usaríamos n

**Interpretação:**
- **std baixo** (0-2) → Frases muito uniformes, estilo monótono
- **std médio** (3-5) → Variação normal
- **std alto** (6+) → Frases muito diversas, estilo irregular

**Diferença humano vs LLM:**
- Δ = **-0.586** (EFEITO GRANDE)
- Humanos: média ~5.2 (mais variável)
- LLMs: média ~3.8 (mais uniforme)

---

### 2.3 sent_burst - Burstiness (Explosividade)

**Conceito:** Mede variação relativa, independente da escala absoluta.

**Fórmula:**

$$
\text{burstiness} = \frac{\sigma}{\mu} = \frac{\text{sent\_std}}{\text{sent\_mean}}
$$

**Código:**

```python
def calcular_sent_burst(text):
    """
    Calcula burstiness = desvio padrão / média.

    Coeficiente de variação para comprimento de frases.
    """
    mean = calcular_sent_mean(text)
    std = calcular_sent_std(text)

    if mean == 0:
        return 0.0

    burst = std / mean
    return burst

# Exemplo comparativo:
# Texto 1: frases de 10±2 palavras
texto1 = "..." # frases: [8, 9, 10, 11, 12]
# mean=10, std=1.58, burst=0.158

# Texto 2: frases de 10±5 palavras
texto2 = "..." # frases: [5, 7, 10, 13, 15]
# mean=10, std=4.18, burst=0.418

# Texto 3: frases de 5±1 palavras
texto3 = "..." # frases: [4, 4.5, 5, 5.5, 6]
# mean=5, std=0.79, burst=0.158

# Observe: texto1 e texto3 têm MESMO burst mas diferentes médias!
# Burstiness captura variação RELATIVA, não absoluta.
```

**Por que burstiness é importante?**

Imagine dois autores:
- **Autor A:** Frases de 20±10 palavras (muito longas, variação grande)
- **Autor B:** Frases de 10±5 palavras (médias, variação média)

- Autor A: std=10 (maior), mas burst=10/20=0.5
- Autor B: std=5 (menor), mas burst=5/10=0.5

**Burstiness = 0.5 em ambos** → Proporcionalmente, variam igualmente!

**Interpretação:**
- **burst baixo** (0.1-0.3) → Regularidade, previsibilidade
- **burst alto** (0.5+) → Irregularidade, "explosividade"

**Diferença humano vs LLM:**
- Δ = **-0.599** (EFEITO GRANDE)
- Humanos: média ~0.48 (mais "explosivos")
- LLMs: média ~0.33 (mais regulares)

**Leitura aprofundada:**
- **Madsen, Kauchak & Elkan (2005)** - "Modeling word burstiness using the Dirichlet distribution"
  - ICML 2005, pp. 545-552
  - Explica burstiness em contexto de modelagem de tópicos

---

## 3. Diversidade Lexical (ttr, herdan_c, hapax_prop)

### 3.1 ttr - Type-Token Ratio

**Conceitos básicos:**
- **Token:** Cada ocorrência de uma palavra (com repetições)
- **Type:** Palavra única (sem repetições)

```python
def calcular_ttr(text):
    """
    Calcula Type-Token Ratio (relação tipo-token).

    TTR = (número de palavras únicas) / (total de palavras)

    Retorna:
        float: Valor entre 0 e 1
    """
    tokens = tokenizar_palavras(text)

    if not tokens:
        return 0.0

    # Types: conjunto (set) remove duplicatas
    types = set(tokens)

    ttr = len(types) / len(tokens)
    return ttr

# Exemplo detalhado:
texto = "o gato viu o rato o gato correu"
tokens = ['o', 'gato', 'viu', 'o', 'rato', 'o', 'gato', 'correu']
# Total tokens: 8
# Types: {'o', 'gato', 'viu', 'rato', 'correu'}
# Total types: 5
# TTR = 5/8 = 0.625

print(f"TTR: {calcular_ttr(texto):.3f}")  # 0.625
```

**Casos extremos:**

```python
# Caso 1: Repetição máxima
texto1 = "gato gato gato gato gato"
# Tokens: 5, Types: 1, TTR = 1/5 = 0.2

# Caso 2: Zero repetição
texto2 = "o gato viu um rato preto"
# Tokens: 6, Types: 6, TTR = 6/6 = 1.0

# Caso 3: Texto real
texto3 = """
Era uma vez um gato que vivia em uma casa grande.
O gato adorava caçar ratos, mas era um gato muito preguiçoso.
"""
# Tokens: ~23, Types: ~18, TTR ≈ 0.78
```

**⚠️ PROBLEMA DO TTR: Dependência do comprimento**

```python
# Texto curto
texto_curto = "O gato viu o rato."
# Tokens: 5, Types: 4, TTR = 0.8

# Texto longo (mesma distribuição)
texto_longo = "O gato viu o rato. " * 100  # Repetir 100×
# Tokens: 500, Types: 4, TTR = 0.008 (!!!)

# TTR CAIS DRASTICAMENTE apenas porque o texto é mais longo!
```

**Lei de Heaps:** Em textos naturais, o número de types cresce sublinearmente com o número de tokens:

$$
V \approx K \cdot N^\beta
$$

Onde:
- V = número de types
- N = número de tokens
- K, β = constantes (tipicamente β ≈ 0.4-0.6)

**Isso significa:** Textos longos SEMPRE terão TTR menor, mesmo se escritos pelo mesmo autor!

**Como lidar com isso?**
1. **Normalizar comprimentos** (truncar ou amostrar)
2. **Usar medidas alternativas** como MTLD, MATTR, HD-D
3. **Usar C de Herdan** (próximo item)

**Interpretação prática:**
- **TTR alto** (0.7-0.9) → Vocabulário diverso, pouquíssima repetição
- **TTR médio** (0.5-0.7) → Balanceado
- **TTR baixo** (0.2-0.5) → Vocabulário limitado ou texto longo

**Diferença humano vs LLM:**
- Δ = **+0.636** (EFEITO GRANDE)
- Humanos: mediana ~0.62
- LLMs: mediana ~0.71 (**mais diverso**)

**Por quê?** LLMs têm vocabulário enorme (treinados em bilhões de tokens) e raramente repetem palavras.

**Leitura crítica:**
- **Richards (1987)** - "Type/Token Ratios: what do they really tell us?"
  - Journal of Child Language, 14(2): 201-209
  - **Ponto principal:** TTR é problemático, dependente do comprimento

---

### 3.2 herdan_c - C de Herdan

**Motivação:** Corrigir dependência do comprimento no TTR.

**Fórmula:**

$$
C = \frac{\log V}{\log N}
$$

Onde:
- V = número de types
- N = número de tokens

**Código:**

```python
import math

def calcular_herdan_c(text):
    """
    Calcula o C de Herdan - medida normalizada de diversidade lexical.

    Menos dependente do comprimento que TTR.
    """
    tokens = tokenizar_palavras(text)

    if len(tokens) < 2:
        return 0.0

    types = set(tokens)

    V = len(types)
    N = len(tokens)

    # Evitar log(1) = 0
    if V == 1 or N == 1:
        return 0.0

    herdan_c = math.log(V) / math.log(N)
    return herdan_c

# Exemplo com textos de tamanhos diferentes:
texto_curto = "o gato viu o rato"
# Tokens: 5, Types: 4
# C = log(4)/log(5) = 1.386/1.609 = 0.861

texto_longo = (texto_curto + " ") * 10
# Tokens: 50, Types: 4
# C = log(4)/log(50) = 1.386/3.912 = 0.354

# OBSERVE: C ainda diminui, mas menos drasticamente que TTR
```

**Comparação TTR vs C de Herdan:**

| Comprimento | TTR | C de Herdan |
|-------------|-----|-------------|
| 10 tokens | 0.80 | 0.86 |
| 100 tokens | 0.40 | 0.68 |
| 1000 tokens | 0.20 | 0.54 |

C de Herdan **diminui mais lentamente** com o comprimento.

**Interpretação:**
- **C alto** (0.8-1.0) → Vocabulário extremamente diverso
- **C médio** (0.6-0.8) → Diversidade normal
- **C baixo** (< 0.6) → Vocabulário repetitivo

**Diferença humano vs LLM:**
- Δ = **+0.587** (EFEITO GRANDE)
- Humanos: ~0.71
- LLMs: ~0.78 (mais diverso)

**Padrão similar ao TTR,** confirmando maior diversidade lexical dos LLMs.

**Leitura:**
- **Herdan (1960)** - "Type-token Mathematics"
  - Mouton publishers
  - Fundamentos matemáticos da diversidade lexical

---

### 3.3 hapax_prop - Proporção de Hapax Legomena

**Definição:** "Hapax legomenon" (plural: hapax legomena) = palavra que aparece **exatamente uma vez**.

Do grego: ἅπαξ λεγόμενον = "dito uma vez"

**Código:**

```python
from collections import Counter

def calcular_hapax_prop(text):
    """
    Calcula a proporção de hapax legomena.

    Retorna:
        float: Proporção de palavras que aparecem exatamente 1 vez
    """
    tokens = tokenizar_palavras(text)

    if not tokens:
        return 0.0

    # Contar frequências
    freq = Counter(tokens)

    # Hapax: palavras com frequência = 1
    hapax_count = sum(1 for count in freq.values() if count == 1)

    # Proporção sobre TOTAL DE TOKENS (não types!)
    hapax_prop = hapax_count / len(tokens)

    return hapax_prop

# Exemplo detalhado:
texto = "o gato viu o rato o gato correu rápido"
tokens = ['o', 'gato', 'viu', 'o', 'rato', 'o', 'gato', 'correu', 'rápido']

freq = Counter(tokens)
# Resultado:
# {'o': 3, 'gato': 2, 'viu': 1, 'rato': 1, 'correu': 1, 'rápido': 1}

# Hapax: viu, rato, correu, rápido (4 palavras)
# Total tokens: 9
# hapax_prop = 4/9 = 0.444

print(f"Hapax proportion: {calcular_hapax_prop(texto):.3f}")
```

**⚠️ CUIDADO:** Hapax proportion ≠ (número de hapax) / (número de types)!

**Correto:** hapax_prop = (número de hapax) / (**total de tokens**)

**Por quê?** Queremos saber que **fração do texto** consiste em palavras únicas.

**Interpretação:**
- **hapax_prop alto** (0.6-0.8) → Muitas palavras aparecem só 1 vez (vocabulário rico)
- **hapax_prop médio** (0.4-0.6) → Balanceado
- **hapax_prop baixo** (0.2-0.4) → Muita repetição

**Diferença humano vs LLM:**
- Δ = **+0.613** (EFEITO GRANDE)
- Humanos: ~0.52
- LLMs: ~0.64 (mais hapax)

**Correlação com TTR:** Muito alta (r = 0.87)
- Faz sentido: ambos medem diversidade lexical
- Hapax ≈ "versão extrema" do TTR

**Curiosidade:** Em corpora linguísticos grandes, hapax legomena representam ~40-50% dos types mas apenas ~10-15% dos tokens (Lei de Zipf).

---

## 4. Entropia e Estrutura (char_entropy, func_word_ratio)

### 4.1 char_entropy - Entropia de Shannon no Nível de Caractere

**Conceito:** Medida de "imprevisibilidade" ou "informação" em uma distribuição.

**Fórmula de Shannon:**

$$
H(X) = -\sum_{i=1}^{n} p(x_i) \log_2(p(x_i))
$$

Onde:
- $p(x_i)$ = probabilidade do caractere $x_i$
- Base 2 → entropia em **bits**

**Código completo com explicação:**

```python
from collections import Counter
import math

def calcular_char_entropy(text):
    """
    Calcula entropia de Shannon sobre distribuição de caracteres.

    Retorna:
        float: Entropia em bits
    """
    if not text:
        return 0.0

    # Contar frequência de cada caractere
    freq = Counter(text)
    total = len(text)

    # Calcular entropia
    entropy = 0.0
    for char, count in freq.items():
        # Probabilidade deste caractere
        p = count / total

        # Contribuição para entropia: -p * log2(p)
        entropy -= p * math.log2(p)

    return entropy

# Exemplo 1: String uniforme (máxima entropia)
texto1 = "abcd"
# Cada caractere aparece 1×, probabilidade = 0.25
# H = -4 × (0.25 × log2(0.25))
#   = -4 × (0.25 × -2)
#   = -4 × -0.5
#   = 2.0 bits

print(f"Entropia (uniforme): {calcular_char_entropy(texto1):.2f}")  # 2.0

# Exemplo 2: String concentrada (baixa entropia)
texto2 = "aaab"
# 'a': 3× (p=0.75), 'b': 1× (p=0.25)
# H = -(0.75 × log2(0.75) + 0.25 × log2(0.25))
#   = -(0.75 × -0.415 + 0.25 × -2)
#   = -(-0.311 - 0.5)
#   = 0.811 bits

print(f"Entropia (concentrada): {calcular_char_entropy(texto2):.2f}")  # 0.81

# Exemplo 3: Caso extremo - repetição total
texto3 = "aaaa"
# 'a': 4× (p=1.0)
# H = -(1.0 × log2(1.0)) = -(1.0 × 0) = 0 bits

print(f"Entropia (total repetição): {calcular_char_entropy(texto3):.2f}")  # 0.0
```

**Intuição: O Que Entropia Realmente Significa?**

Entropia = "Quantos bits (em média) preciso para codificar cada caractere?"

- **Texto previsível** (baixa entropia) → Posso comprimir muito (ex: "aaaa" → "4a")
- **Texto imprevisível** (alta entropia) → Difícil comprimir

**Exemplo prático com texto real:**

```python
texto_pt = "O gato está dormindo pacificamente na cama."
entropy_pt = calcular_char_entropy(texto_pt)
print(f"Entropia (português): {entropy_pt:.2f} bits")
# Resultado típico: ~4.2-4.6 bits

# Por que não é máximo?
# Máximo teórico = log2(alfabeto)
# Português tem ~26 letras + espaço + pontuação ≈ 40 símbolos
# Máximo = log2(40) = 5.32 bits
# Mas distribuição NÃO é uniforme:
#   - Vogais mais comuns que consoantes
#   - 'a', 'e', 'o' muito frequentes
#   - 'k', 'w', 'y' raros
```

**Diferença humano vs LLM:**
- Δ = **-0.881** (EFEITO **MUITO GRANDE**, o maior de todos!)
- Humanos: mediana ~4.58 bits (**maior entropia**)
- LLMs: mediana ~4.12 bits (menor entropia)

**Por que humanos têm entropia MAIOR?**

1. **Erros de digitação:**
   ```
   Humano: "O gaot está dromindo."  # Typos aumentam caracteres raros
   LLM: "O gato está dormindo."     # Sempre correto
   ```

2. **Pontuação variada:**
   ```
   Humano: "Sério?! Não acredito..."
   LLM: "Sério? Não acredito."
   ```

3. **Uso de símbolos/emojis:**
   ```
   Humano: "Adorei! :) 10/10"
   LLM: "Adorei muito."
   ```

4. **Variação ortográfica:**
   ```
   Humano: "tbm", "vc", "pq"
   LLM: "também", "você", "porque"
   ```

**Leitura fundamental:**
- **Shannon (1948)** - "A Mathematical Theory of Communication"
  - Bell System Technical Journal, 27(3): 379-423
  - Paper FUNDACIONAL da teoria da informação
  - Define entropia, compressão, capacidade de canal

---

### 4.2 func_word_ratio - Proporção de Palavras Funcionais

**Definição:** Palavras funcionais (ou gramaticais) = palavras sem conteúdo semântico próprio.

**Categorias:**
1. **Artigos:** o, a, os, as, um, uma, uns, umas
2. **Preposições:** de, em, para, com, por, sem, sobre, entre, até, desde
3. **Pronomes:** eu, tu, ele, ela, nós, vós, eles, elas, me, te, se, lhe, meu, seu, este, esse, aquele
4. **Conjunções:** e, ou, mas, porque, que, se, quando, embora, portanto
5. **Advérbios comuns:** não, sim, muito, mais, menos, já, ainda, sempre, nunca
6. **Verbos auxiliares:** ser, estar, ter, haver (formas conjugadas)

**Contraste: Palavras de Conteúdo**
- Substantivos: gato, casa, amor, computador
- Verbos principais: correr, pensar, escrever
- Adjetivos: bonito, rápido, inteligente
- Advérbios de modo: rapidamente, calmamente

**Código com lista completa:**

```python
# Lista de palavras funcionais em português (exemplo parcial)
FUNC_WORDS_PT = {
    # Artigos
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',

    # Preposições
    'de', 'em', 'para', 'com', 'por', 'sem', 'sob', 'sobre',
    'entre', 'até', 'desde', 'perante', 'após', 'contra',

    # Pronomes
    'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas',
    'me', 'te', 'se', 'lhe', 'nos', 'vos', 'lhes',
    'meu', 'minha', 'meus', 'minhas', 'teu', 'tua', 'seu', 'sua',
    'nosso', 'nossa', 'vosso', 'vossa',
    'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas',
    'aquele', 'aquela', 'aqueles', 'aquelas',
    'isto', 'isso', 'aquilo',
    'que', 'quem', 'qual', 'quais', 'quanto', 'quantos',

    # Conjunções
    'e', 'ou', 'mas', 'porém', 'contudo', 'todavia', 'entretanto',
    'porque', 'pois', 'que', 'se', 'caso', 'quando', 'como',
    'embora', 'conquanto', 'ainda', 'logo', 'portanto',

    # Advérbios comuns
    'não', 'sim', 'muito', 'pouco', 'mais', 'menos',
    'já', 'ainda', 'sempre', 'nunca', 'jamais',
    'aqui', 'aí', 'ali', 'lá', 'cá',
    'hoje', 'ontem', 'amanhã', 'agora', 'depois', 'antes',

    # Formas verbais auxiliares comuns
    'é', 'são', 'foi', 'foram', 'será', 'serão', 'seja', 'sejam',
    'era', 'eram', 'sendo', 'sido',
    'está', 'estão', 'estava', 'estavam', 'estando', 'estado',
    'tem', 'têm', 'tinha', 'tinham', 'tendo', 'tido',
    'há', 'havia', 'havendo', 'havido',
}

def calcular_func_word_ratio(text):
    """
    Calcula proporção de palavras funcionais.

    Retorna:
        float: Razão entre palavras funcionais e total de palavras
    """
    tokens = tokenizar_palavras(text)

    if not tokens:
        return 0.0

    # Contar palavras funcionais
    func_count = sum(1 for token in tokens if token in FUNC_WORDS_PT)

    func_ratio = func_count / len(tokens)
    return func_ratio

# Exemplo 1: Texto com muitas palavras funcionais
texto1 = "Eu não sabia que ele era o dono da casa."
# Funcionais: eu, não, que, ele, era, o, da (7/10 = 0.7)
print(f"Func ratio 1: {calcular_func_word_ratio(texto1):.2f}")

# Exemplo 2: Texto com poucas palavras funcionais
texto2 = "Gato preto pulou muro alto."
# Funcionais: (nenhuma!) (0/5 = 0.0)
print(f"Func ratio 2: {calcular_func_word_ratio(texto2):.2f}")

# Exemplo 3: Texto normal
texto3 = "O gato preto dormia tranquilamente sob a árvore grande."
# Funcionais: o, sob, a (3/9 = 0.33)
print(f"Func ratio 3: {calcular_func_word_ratio(texto3):.2f}")
```

**Interpretação:**
- **func_word_ratio alto** (0.5-0.7) → Texto gramaticalmente rico, formal
- **func_word_ratio médio** (0.3-0.5) → Normal
- **func_word_ratio baixo** (0.1-0.3) → Telegráfico, substantivos/verbos dominam

**Diferença humano vs LLM:**
- Δ = **+0.361** (EFEITO MÉDIO)
- Humanos: ~0.38
- LLMs: ~0.43 (usam mais palavras funcionais)

**Hipótese:** LLMs treinados em textos formais (Wikipedia, livros, artigos) aprendem estrutura gramatical explícita com mais preposições, artigos, conjunções.

**Uso histórico:**
- **Mosteller & Wallace (1964)** - Usaram palavras funcionais para atribuir autoria dos Federalist Papers
  - Descobriram que "by", "to", "of" distinguem autores melhor que palavras de conteúdo!
  - Palavras funcionais = "impressão digital" do autor

**Leitura:**
- **Eder (2015)** - "Does size matter? Authorship attribution, small samples, big problem"
  - Digital Scholarship in the Humanities, 30(2): 167-182
  - Mostra que palavras funcionais são mais robustas que palavras de conteúdo

---

### 4.2 first_person_ratio - Proporção de Pronomes de Primeira Pessoa

**Por quê?** Mede o grau de "pessoalidade" e subjetividade do texto.

**Implementação:**

```python
import re
from collections import Counter

# Lista de pronomes de primeira pessoa em português
FIRST_PERSON_PRONOUNS = {
    'eu', 'me', 'mim', 'comigo',
    'nós', 'nos', 'conosco',
    'meu', 'minha', 'meus', 'minhas',
    'nosso', 'nossa', 'nossos', 'nossas'
}

def calcular_first_person_ratio(text):
    """
    Calcula proporção de pronomes de primeira pessoa.

    Retorna:
        float: Razão de pronomes de primeira pessoa sobre total de tokens
    """
    # Tokenizar
    tokens = word_tokenize(text.lower(), language='portuguese')

    if not tokens:
        return 0.0

    # Contar pronomes de primeira pessoa
    first_person_count = sum(1 for token in tokens if token in FIRST_PERSON_PRONOUNS)

    ratio = first_person_count / len(tokens)
    return ratio

# Exemplo 1: Texto altamente pessoal
texto1 = "Eu não sei se minha opinião importa, mas eu acho que nós deveríamos reconsiderar."
# Pronomes: eu (2x), minha (1x), nós (1x) = 4/14 = 0.286
print(f"First person ratio 1: {calcular_first_person_ratio(texto1):.3f}")

# Exemplo 2: Texto impessoal/objetivo
texto2 = "O sistema foi projetado para processar dados em tempo real."
# Pronomes: (nenhum!) = 0/10 = 0.0
print(f"First person ratio 2: {calcular_first_person_ratio(texto2):.3f}")

# Exemplo 3: Narrativa pessoal
texto3 = "Meu pai me disse que eu deveria estudar mais. Nossos professores concordaram."
# Pronomes: meu, me, eu, nossos = 4/13 = 0.308
print(f"First person ratio 3: {calcular_first_person_ratio(texto3):.3f}")
```

**Interpretação:**
- **first_person_ratio alto** (0.1-0.3) → Texto pessoal, narrativo, subjetivo
- **first_person_ratio médio** (0.05-0.1) → Balanceado
- **first_person_ratio baixo** (0.0-0.05) → Texto objetivo, impessoal, técnico

**Diferença humano vs LLM:**
- Δ = **-0.424** (EFEITO MÉDIO, negativo)
- Humanos: mediana ~0.038
- LLMs: mediana ~0.022 (**menos pessoal**)

**Por quê?** LLMs tendem a gerar textos mais impessoais e objetivos. Quando humanos escrevem, frequentemente incluem opiniões pessoais ("eu acho", "na minha opinião", "nós observamos").

**Observação importante:** Esta métrica é **altamente dependente do domínio**:
- **Blogs/Redes sociais:** First person ratio ALTO esperado
- **Artigos científicos:** First person ratio BAIXO esperado (exceto "nós observamos")
- **Notícias:** First person ratio BAIXO

**Extensões possíveis:**
1. Separar singular (eu, meu) de plural (nós, nosso)
2. Incluir segunda pessoa (você, tu) e terceira pessoa
3. Calcular "Person Shift Rate" (mudanças entre pessoas gramaticais)

**Leitura:**
- **Pennebaker & King (1999)** - "Linguistic Styles: Language Use as an Individual Difference"
  - Journal of Personality and Social Psychology, 77(6): 1296-1312
  - Mostra que uso de pronomes revela traços de personalidade

---

### 4.3 bigram_repeat_ratio - Proporção de Bigramas Repetidos

**Por quê?** Mede redundância lexical e tendência à repetição de padrões.

**Conceito de Bigrama:**
Um bigrama é um par consecutivo de palavras. Para "o gato preto", os bigramas são:
- ("o", "gato")
- ("gato", "preto")

**Implementação:**

```python
from collections import Counter

def calcular_bigram_repeat_ratio(text):
    """
    Calcula proporção de bigramas repetidos.

    Bigrama repetido = bigrama que aparece 2+ vezes no texto.

    Retorna:
        float: Razão de bigramas únicos que aparecem 2+ vezes
    """
    # Tokenizar
    tokens = word_tokenize(text.lower(), language='portuguese')

    if len(tokens) < 2:
        return 0.0

    # Gerar bigramas
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

    if not bigrams:
        return 0.0

    # Contar ocorrências de cada bigrama
    bigram_counts = Counter(bigrams)

    # Contar quantos bigramas aparecem 2+ vezes
    repeated_bigrams = sum(1 for count in bigram_counts.values() if count >= 2)

    # Total de bigramas únicos
    unique_bigrams = len(bigram_counts)

    ratio = repeated_bigrams / unique_bigrams if unique_bigrams > 0 else 0.0
    return ratio

# Exemplo 1: Texto com repetições
texto1 = "O gato preto perseguiu o gato branco. O gato preto venceu."
# Bigramas:
# ("o", "gato") - 3x (REPETIDO)
# ("gato", "preto") - 2x (REPETIDO)
# ("preto", "perseguiu") - 1x
# ("perseguiu", "o") - 1x
# ("gato", "branco") - 1x
# ("branco", "o") - 1x (considera "branco" + "." + "o")
# ("preto", "venceu") - 1x
# Total: 7 bigramas únicos, 2 repetidos → 2/7 = 0.286
print(f"Bigram repeat ratio 1: {calcular_bigram_repeat_ratio(texto1):.3f}")

# Exemplo 2: Texto sem repetições
texto2 = "Cada palavra aparece exatamente uma única vez nesta sentença criativa."
# Todos bigramas únicos → 0/9 = 0.0
print(f"Bigram repeat ratio 2: {calcular_bigram_repeat_ratio(texto2):.3f}")

# Exemplo 3: Texto com muitas repetições
texto3 = "Sim, sim, sim! Não, não, não! Talvez, talvez, talvez!"
# ("sim", "sim") - 2x (REPETIDO)
# ("não", "não") - 2x (REPETIDO)
# ("talvez", "talvez") - 2x (REPETIDO)
# Total: 6 bigramas únicos, 3 repetidos → 3/6 = 0.5
print(f"Bigram repeat ratio 3: {calcular_bigram_repeat_ratio(texto3):.3f}")
```

**Interpretação:**
- **bigram_repeat_ratio alto** (0.3-0.6) → Texto repetitivo, redundante
- **bigram_repeat_ratio médio** (0.1-0.3) → Normal
- **bigram_repeat_ratio baixo** (0.0-0.1) → Texto altamente diverso, pouca repetição

**Diferença humano vs LLM:**
- Δ = **-0.147** (EFEITO PEQUENO, negativo)
- Humanos: mediana ~0.12
- LLMs: mediana ~0.10 (**menos repetitivo**)

**Por quê?** LLMs modernos (especialmente com nucleus sampling, temperatura ajustada) evitam ativamente repetições através de mecanismos de penalização durante geração. Humanos, por outro lado, repetem naturalmente estruturas sintáticas ("o gato... o gato...").

**Variações da métrica:**
1. **Trigrama repeat ratio** (3 palavras consecutivas)
2. **Skip-bigram repeat ratio** (pares não consecutivos)
3. **Weighted repeat ratio** (ponderar por frequência: bigramas que aparecem 10x vs 2x)

**Conexão com Burstiness:**
- **sent_burst** mede variação no COMPRIMENTO de frases
- **bigram_repeat_ratio** mede REPETIÇÃO LEXICAL
- Ambos capturam aspectos de "diversidade" textual

**Cuidado:** Textos técnicos (código, fórmulas, listas) naturalmente têm ALTO bigram repeat ratio:
```
Passo 1: Abra o arquivo.
Passo 2: Leia o arquivo.
Passo 3: Feche o arquivo.
```
→ ("passo", "1"), ("passo", "2"), ("passo", "3"), ("o", "arquivo") todos repetidos!

**Leitura:**
- **Madsen et al. (2005)** - "Modeling word burstiness using the Dirichlet distribution"
  - ICML 2005
  - Modelagem estatística de repetições (palavra-nível, mas conceito estende para bigramas)

---

## 5. Resumo das 10 Características

| # | Nome | O que mede | Efeito (Δ) | Direção |
|---|------|-----------|-----------|---------|
| 1 | sent_mean | Comprimento médio de frases | +0.349 | LLM > Humano |
| 2 | sent_std | Variabilidade do comprimento | -0.364 | Humano > LLM |
| 3 | sent_burst | Burstiness (coef. variação) | -0.453 | Humano > LLM |
| 4 | ttr | Diversidade lexical básica | +0.636 | LLM > Humano |
| 5 | herdan_c | Diversidade normalizada | +0.616 | LLM > Humano |
| 6 | hapax_prop | Palavras raras | +0.555 | LLM > Humano |
| 7 | char_entropy | Uniformidade de caracteres | +0.173 | LLM > Humano |
| 8 | func_word_ratio | Palavras funcionais | +0.361 | LLM > Humano |
| 9 | first_person_ratio | Pronomes 1ª pessoa | -0.424 | Humano > LLM |
| 10 | bigram_repeat_ratio | Repetição de bigramas | -0.147 | Humano > LLM |

**Padrões-chave identificados:**

### LLMs tendem a:
1. ✅ Escrever frases **mais longas** e **uniformes** (sent_mean ↑, sent_std ↓, sent_burst ↓)
2. ✅ Usar vocabulário **mais diverso** (ttr ↑, herdan_c ↑, hapax_prop ↑)
3. ✅ Usar mais **palavras funcionais** (func_word_ratio ↑)
4. ✅ Ser mais **impessoais** (first_person_ratio ↓)
5. ✅ Evitar **repetições** (bigram_repeat_ratio ↓)

### Humanos tendem a:
1. ✅ Variar comprimento de frases (sent_std ↑, sent_burst ↑)
2. ✅ Repetir vocabulário (ttr ↓)
3. ✅ Usar linguagem **pessoal** (first_person_ratio ↑)
4. ✅ Repetir estruturas (bigram_repeat_ratio ↑)

**Interpretação geral:**
LLMs produzem texto **estatisticamente mais uniforme, formalmente rico, e lexicamente diverso**. Humanos produzem texto **mais variável, pessoal, e com padrões de repetição**.

---

## 6. Próximo Passo

Continue para:
- **[GUIA_03_ESTATISTICA.md](GUIA_03_ESTATISTICA.md)** - Testes estatísticos detalhados (Mann-Whitney U, Cliff's delta, FDR)
