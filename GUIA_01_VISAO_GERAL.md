# Guia Completo do Projeto - Parte 1: Visão Geral

**Público-alvo:** Mestrandos em Ciência da Computação com conhecimento básico de estatística e lógica fuzzy

**Objetivo:** Explicar em detalhes minuciosos cada processo, indicador e método utilizado no projeto de detecção de textos gerados por LLMs em português.

---

## 1. Introdução ao Problema

### 1.1 O Que Estamos Tentando Resolver?

**Problema Central:** Dado um texto em português brasileiro, queremos determinar se ele foi escrito por um humano ou gerado por um modelo de linguagem (LLM) como ChatGPT, GPT-4, etc.

**Por que isso importa?**
- **Educação:** Detectar trabalhos acadêmicos gerados por IA
- **Integridade científica:** Identificar artigos ou seções escritas por LLMs
- **Moderação de conteúdo:** Detectar spam ou desinformação gerada em massa
- **Forense digital:** Atribuição de autoria em investigações

### 1.2 Nossa Abordagem em Duas Frentes

Este projeto usa **duas metodologias complementares** aplicadas aos **mesmos dados**:

#### **Abordagem 1: Análise Estatística** (paper_stat)
- Usa testes estatísticos clássicos (Mann-Whitney U, Cliff's delta)
- Aplica modelos de classificação tradicionais (LDA, Regressão Logística)
- **Vantagens:** Alta precisão (97% AUC), rigor matemático estabelecido
- **Desvantagens:** Menos interpretável ("caixa preta")

#### **Abordagem 2: Lógica Fuzzy** (paper_fuzzy)
- Usa conjuntos fuzzy e funções de pertinência triangulares
- Cria regras interpretáveis baseadas em características estilométricas
- **Vantagens:** Totalmente interpretável, robustez excepcional (variância 3-4× menor)
- **Desvantagens:** Menor precisão (89% AUC) - "custo da interpretabilidade"

**Importante:** Ambas as abordagens usam **exatamente as mesmas 10 características estilométricas** extraídas do texto. A diferença está em **como** essas características são usadas para classificação.

---

## 2. Pipeline Geral do Projeto

```
┌─────────────────────────────────────────────────────────────────┐
│                    ETAPA 1: COLETA DE DADOS                     │
├─────────────────────────────────────────────────────────────────┤
│ • Corpus original: 1.295.958 textos (98% humano, 2% LLM)      │
│ • Fontes: BrWaC, BoolQ, ShareGPT-PT, Canarim, IMDB traduções   │
│ • Resultado: arquivo balanced.csv (3.2 GB)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                ETAPA 2: AMOSTRAGEM E BALANCEAMENTO              │
├─────────────────────────────────────────────────────────────────┤
│ • Estratificação: 50.000 humano + 50.000 LLM = 100.000 total  │
│ • Método: downsampling (maioria) + upsampling (minoria)       │
│ • Seed: 42 (para reprodutibilidade)                           │
│ • Resultado: balanced_sample_100k.csv (257 MB)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              ETAPA 3: EXTRAÇÃO DE CARACTERÍSTICAS               │
├─────────────────────────────────────────────────────────────────┤
│ • Código: src/features.py                                      │
│ • Entrada: texto bruto em português                           │
│ • Saída: 10 métricas numéricas por texto                      │
│ • Resultado: features_100k.csv (17 MB)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            ETAPA 4A: ANÁLISE ESTATÍSTICA (paper_stat)           │
├─────────────────────────────────────────────────────────────────┤
│ • Testes de hipótese: Mann-Whitney U                          │
│ • Tamanho de efeito: Cliff's delta                            │
│ • Correção para múltiplos testes: Benjamini-Hochberg (FDR)    │
│ • Análise multivariada: PCA                                   │
│ • Classificação: LDA, Regressão Logística                     │
│ • Validação: 5-fold stratified cross-validation               │
│ • Resultado: 97.03% AUC                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             ETAPA 4B: CLASSIFICAÇÃO FUZZY (paper_fuzzy)         │
├─────────────────────────────────────────────────────────────────┤
│ • Funções de pertinência: triangulares baseadas em quantis    │
│ • Conjuntos fuzzy: "baixo", "médio", "alto" para cada feature │
│ • Sistema de inferência: Takagi-Sugeno ordem zero             │
│ • Agregação: média aritmética simples                         │
│ • Validação: 5-fold stratified cross-validation               │
│ • Resultado: 89.34% AUC (±0.04% std)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ETAPA 5: VISUALIZAÇÃO                         │
├─────────────────────────────────────────────────────────────────┤
│ • Boxplots das 10 características (humano vs LLM)             │
│ • Scatter plot PCA (PC1 vs PC2)                               │
│ • Matriz de correlação (heatmap)                              │
│ • Curvas ROC (comparando 3 classificadores)                   │
│ • Curvas Precision-Recall                                     │
│ • Funções de pertinência fuzzy (10 características × 3 sets)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. As 10 Características Estilométricas

**Estilometria** = análise quantitativa do estilo de escrita. Cada texto é transformado em 10 números que capturam diferentes aspectos do estilo.

### 3.1 Grupo 1: Estatísticas de Frase (3 features)

#### **sent_mean** - Comprimento médio de frase
- **O que é:** Número médio de palavras por frase
- **Como calcular:** (total de palavras) ÷ (número de frases)
- **Exemplo:** "Eu gosto de café. Meu irmão prefere chá." → 2 frases, 9 palavras → 4.5 palavras/frase
- **Padrão observado:** LLMs tendem a ter frases ligeiramente mais longas e uniformes

#### **sent_std** - Desvio padrão do comprimento de frase
- **O que é:** Medida de variabilidade no tamanho das frases
- **Como calcular:** Desvio padrão dos comprimentos de todas as frases
- **Interpretação:**
  - `sent_std` alto = frases de tamanhos muito variados (textos humanos)
  - `sent_std` baixo = frases de tamanho uniforme (textos de LLM)
- **Padrão observado:** Humanos variam mais (δ = -0.586, efeito grande)

#### **sent_burst** - Burstiness (explosividade)
- **O que é:** Razão entre desvio padrão e média: `sent_burst = sent_std / sent_mean`
- **Interpretação:**
  - `sent_burst` alto = muita variação relativa (ex: frases de 3 e 30 palavras)
  - `sent_burst` baixo = pouca variação relativa (ex: frases de 8 a 12 palavras)
- **Por que importa:** Captura irregularidade estrutural independente da escala
- **Padrão observado:** Humanos mais "explosivos" (δ = -0.599, efeito grande)
- **Leitura sugerida:** Madsen et al. (2005) - "Modeling word burstiness using the Dirichlet distribution"

---

### 3.2 Grupo 2: Diversidade Lexical (3 features)

#### **ttr** - Type-Token Ratio (Relação Tipo-Token)
- **O que é:** Razão entre palavras únicas e total de palavras
- **Como calcular:** `TTR = (número de palavras únicas) ÷ (total de palavras)`
- **Exemplo:** "O gato viu o rato" → 5 palavras, 4 únicas (gato, viu, rato, o) → TTR = 4/5 = 0.8
- **Interpretação:**
  - TTR alto = vocabulário diverso, pouca repetição
  - TTR baixo = vocabulário limitado, muita repetição
- **⚠️ LIMITAÇÃO CONHECIDA:** TTR depende do comprimento do texto! Textos mais longos naturalmente têm TTR menor.
- **Padrão observado:** LLMs têm TTR mais alto (δ = +0.636, efeito grande)
- **Leitura sugerida:**
  - Richards (1987) - "Type/Token Ratios: what do they really tell us?" - Crítica ao TTR
  - McCarthy & Jarvis (2010) - "MTLD, vocd-D, and HD-D" - Alternativas melhores

#### **herdan_c** - C de Herdan
- **O que é:** Versão normalizada do TTR que reduz dependência do comprimento
- **Fórmula:** `C = log(V) / log(N)` onde V = vocabulário, N = tokens
- **Como calcular:**
  ```python
  import math
  V = len(set(words))  # palavras únicas
  N = len(words)       # total de palavras
  herdan_c = math.log(V) / math.log(N)
  ```
- **Interpretação:** Similar ao TTR, mas mais robusto para textos de tamanhos diferentes
- **Padrão observado:** LLMs têm C maior (δ = +0.587)
- **Leitura sugerida:** Herdan (1960) - "Type-token Mathematics"

#### **hapax_prop** - Proporção de Hapax Legomena
- **O que são hapax legomena:** Palavras que aparecem exatamente uma vez no texto
- **Como calcular:** `hapax_prop = (palavras com frequência = 1) ÷ (total de palavras)`
- **Exemplo:** "O gato viu o rato preto" → 6 palavras
  - Frequências: o=2, gato=1, viu=1, rato=1, preto=1
  - Hapax: gato, viu, rato, preto (4 palavras)
  - hapax_prop = 4/6 = 0.667
- **Interpretação:** Mede originalidade e diversidade vocabular
- **Padrão observado:** LLMs têm mais hapax (δ = +0.613) - vocabulário mais diverso
- **Correlação:** Hapax e TTR são fortemente correlacionados (r = 0.87)

---

### 3.3 Grupo 3: Entropia e Estrutura (2 features)

#### **char_entropy** - Entropia de Shannon em nível de caractere
- **O que é:** Medida de "surpresa" ou imprevisibilidade na distribuição de caracteres
- **Fórmula:** `H = -Σ p(c) × log₂(p(c))` para cada caractere c
- **Como calcular:**
  ```python
  from collections import Counter
  import math

  text = "exemplo de texto"
  freq = Counter(text)
  total = len(text)

  entropy = 0
  for char, count in freq.items():
      p = count / total
      entropy -= p * math.log2(p)
  ```
- **Interpretação:**
  - Entropia alta = distribuição uniforme, imprevisível (ex: "abcdefgh")
  - Entropia baixa = distribuição concentrada, previsível (ex: "aaaabbbb")
- **Exemplo prático:**
  - "aaaa" → entropia ≈ 0 (totalmente previsível)
  - "abcd" → entropia = 2.0 (máxima para 4 símbolos)
  - Texto real português → entropia ≈ 4.2-4.6 bits
- **Padrão observado:** Humanos têm entropia MAIOR (δ = -0.881, **EFEITO MAIOR DE TODOS**)
  - **Por quê?** Humanos cometem erros de digitação, usam pontuação variada, incluem emojis, acentos irregulares
  - LLMs geram texto "limpo" e previsível
- **Leitura sugerida:** Shannon (1948) - "A Mathematical Theory of Communication" (paper fundacional da teoria da informação)

#### **func_word_ratio** - Proporção de palavras funcionais
- **O que são palavras funcionais:** Palavras sem conteúdo semântico próprio (artigos, preposições, pronomes, conjunções)
- **Exemplos em português:** "o", "a", "de", "em", "que", "para", "com", "por", "se", "não", "mas", "como"
- **Como calcular:** `func_word_ratio = (palavras funcionais) ÷ (total de palavras)`
- **Lista usada no projeto:** ~200 palavras funcionais mais comuns em português
  ```python
  func_words = {"o", "a", "os", "as", "um", "uma", "de", "em", "para",
                "com", "por", "que", "se", "não", "mais", ...}
  ```
- **Interpretação:** Mede densidade de estrutura gramatical vs conteúdo
- **Padrão observado:** LLMs usam mais palavras funcionais (δ = +0.361, efeito médio)
  - **Hipótese:** LLMs treinados em textos formais tendem a estrutura gramatical explícita
- **Leitura sugerida:**
  - Mosteller & Wallace (1964) - "Inference and Disputed Authorship: The Federalist" - Uso pioneiro de palavras funcionais
  - Eder (2015) - "Does size matter? Authorship attribution, small samples, big problem"

---

### 3.4 Grupo 4: Características de Autoria (2 features)

#### **first_person_ratio** - Proporção de pronomes de primeira pessoa
- **O que é:** Fração de palavras que são pronomes de 1ª pessoa
- **Pronomes incluídos:** "eu", "me", "mim", "comigo", "nós", "nos", "conosco", "meu", "minha", "meus", "minhas", "nosso", "nossa", etc.
- **Como calcular:**
  ```python
  first_person = {"eu", "me", "mim", "comigo", "nós", "nos", "conosco",
                  "meu", "minha", "meus", "minhas", "nosso", "nossa", ...}
  first_person_ratio = sum(1 for word in tokens if word in first_person) / len(tokens)
  ```
- **Interpretação:** Textos mais pessoais/narrativos vs objetivos/descritivos
- **Padrão observado:** Efeito **negligenciável** (δ = -0.049, p < 0.001 mas irrelevante na prática)
  - **Por quê?** Ambos corpus contêm mix de textos pessoais e objetivos
  - Significância estatística (p-valor) ≠ significância prática (tamanho de efeito)

#### **bigram_repeat_ratio** - Proporção de bigramas repetidos
- **O que são bigramas:** Pares consecutivos de palavras
- **Exemplo:** "O gato viu o rato" → bigramas: ["O gato", "gato viu", "viu o", "o rato"]
- **Como calcular:**
  ```python
  bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
  unique_bigrams = len(set(bigrams))
  total_bigrams = len(bigrams)
  bigram_repeat_ratio = 1 - (unique_bigrams / total_bigrams)
  ```
- **Interpretação:**
  - Ratio alto = muitos bigramas repetidos (textos repetitivos)
  - Ratio baixo = poucos bigramas repetidos (textos diversos)
- **Padrão observado:** Humanos repetem menos bigramas (δ = -0.231, efeito pequeno)

---

## 4. Tamanhos de Efeito: Interpretação

Nos testes estatísticos, usamos **Cliff's delta (δ)** como medida de tamanho de efeito.

### 4.1 O Que É Cliff's Delta?

**Definição intuitiva:** Se escolhermos aleatoriamente um texto humano e um texto de LLM, qual a probabilidade de que uma característica seja maior no humano menos a probabilidade de ser maior no LLM?

**Fórmula conceitual:**
```
δ = P(X_humano > X_llm) - P(X_humano < X_llm)
```

**Escala de interpretação** (Romano et al. 2006):
- `|δ| < 0.147` → Efeito **negligenciável** (diferença irrelevante)
- `0.147 ≤ |δ| < 0.330` → Efeito **pequeno** (diferença detectável mas sutil)
- `0.330 ≤ |δ| < 0.474` → Efeito **médio** (diferença clara)
- `|δ| ≥ 0.474` → Efeito **grande** (diferença substancial)

**Sinal:**
- δ **negativo** → LLMs tendem a ter valores **maiores**
- δ **positivo** → Humanos tendem a ter valores **maiores**

### 4.2 Resultados do Nosso Projeto

| Característica | Delta (δ) | Tamanho de Efeito | Interpretação |
|----------------|-----------|-------------------|---------------|
| char_entropy | **-0.881** | GRANDE | LLMs têm entropia muito maior |
| sent_burst | **-0.599** | GRANDE | Humanos mais "explosivos" |
| ttr | **+0.636** | GRANDE | LLMs mais diversos lexicalmente |
| sent_std | **-0.586** | GRANDE | Humanos variam mais |
| herdan_c | **+0.587** | GRANDE | LLMs vocabulário maior |
| hapax_prop | **+0.613** | GRANDE | LLMs mais palavras únicas |
| func_word_ratio | **+0.361** | MÉDIO | LLMs usam mais palavras funcionais |
| bigram_repeat_ratio | **-0.231** | PEQUENO | Humanos repetem menos |
| sent_mean | **+0.126** | NEGLIGENCIÁVEL | Praticamente igual |
| first_person_ratio | **-0.049** | NEGLIGENCIÁVEL | Sem diferença prática |

**Conclusão:** 6 características têm efeito grande, 1 médio, 1 pequeno, 2 negligenciáveis.

---

## 5. Leituras Sugeridas por Tópico

### 5.1 Fundamentos de Estilometria
1. **Mosteller & Wallace (1964)** - "Inference and Disputed Authorship: The Federalist"
   - 📘 Livro clássico, primeiro uso computacional para atribuição de autoria
   - **O que você aprenderá:** Uso de palavras funcionais, método bayesiano

2. **Burrows (2002)** - "'Delta': A Measure of Stylistic Difference"
   - 📄 Paper curto (20 páginas), altamente citado
   - **O que você aprenderá:** Medida Delta para comparar estilos, base da estilometria moderna

3. **Stamatatos (2009)** - "A survey of modern authorship attribution methods"
   - 📄 Survey completo (20 páginas), excelente panorama
   - **O que você aprenderá:** Todos os métodos de atribuição de autoria até 2009

### 5.2 Detecção de LLMs (Recente, 2023-2025)
4. **Herbold et al. (2023)** - "A Large-Scale Comparison of Human-Written Versus ChatGPT-Generated Essays"
   - 📄 Scientific Data (Nature), peer-reviewed
   - **O que você aprenderá:** 31 características, Random Forest, 81-98% acurácia

5. **Zaitsu & Jin (2023)** - "Distinguishing ChatGPT-generated and human-written papers through Japanese stylometric analysis"
   - 📄 PLOS One, aplicação em japonês
   - **O que você aprenderá:** Validação cross-linguistic, 100% precisão

6. **Przystalski et al. (2025)** - "Stylometry recognizes human and LLM-generated texts in short samples"
   - 📄 Expert Systems with Applications, mais recente
   - **O que você aprenderá:** StyloMetrix, centenas de features, performance em textos curtos

### 5.3 Estatística Não-Paramétrica
7. **Mann & Whitney (1947)** - "On a test of whether one of two random variables is stochastically larger"
   - 📄 Paper original do teste U, matemática pesada
   - **Alternativa mais acessível:** Siegel & Castellan (1988) - Livro didático

8. **Cliff (1993)** - "Dominance statistics: Ordinal analyses to answer ordinal questions"
   - 📄 Psychological Bulletin, introduz Cliff's delta
   - **O que você aprenderá:** Por que delta é melhor que Cohen's d para dados ordinais

9. **Benjamini & Hochberg (1995)** - "Controlling the false discovery rate"
   - 📄 JRSS, correção FDR (mais liberal que Bonferroni)
   - **O que você aprenderá:** Correção para múltiplos testes, controle de FDR

### 5.4 Lógica Fuzzy
10. **Zadeh (1965)** - "Fuzzy sets"
    - 📄 Information and Control, paper fundacional
    - **O que você aprenderá:** Conceito de pertinência gradual, funções de pertinência

11. **Klir & Yuan (1995)** - "Fuzzy Sets and Fuzzy Logic: Theory and Applications"
    - 📘 Livro-texto completo (574 páginas), referência definitiva
    - **O que você aprenderá:** Tudo sobre fuzzy - teoria, aplicações, exemplos

12. **Pedrycz (1994)** - "Why triangular membership functions?"
    - 📄 Fuzzy Sets and Systems, justificativa teórica
    - **O que você aprenderá:** Por que funções triangulares são boas o suficiente

13. **Ross (2010)** - "Fuzzy Logic with Engineering Applications" (3rd ed.)
    - 📘 Livro didático (600 páginas), foco prático
    - **O que você aprenderá:** Implementação de sistemas fuzzy, exemplos de código

14. **Takagi & Sugeno (1985)** - "Fuzzy identification of systems"
    - 📄 IEEE Trans. SMC, sistemas Takagi-Sugeno
    - **O que você aprenderá:** Sistemas fuzzy com consequentes lineares (ou constantes)

### 5.5 Análise Multivariada
15. **Jolliffe (2002)** - "Principal Component Analysis"
    - 📘 Livro completo sobre PCA (488 páginas)
    - **O que você aprenderá:** Matemática do PCA, interpretação de loadings

16. **Fisher (1936)** - "The use of multiple measurements in taxonomic problems"
    - 📄 Paper original da LDA, exemplo da Iris
    - **O que você aprenderá:** Separação de classes usando projeção linear

17. **Hosmer & Lemeshow (2013)** - "Applied Logistic Regression" (3rd ed.)
    - 📘 Livro didático (500 páginas), padrão-ouro
    - **O que você aprenderá:** Regressão logística detalhada, interpretação de odds ratios

---

## 6. Conceitos a Revisar Antes de Ler os Papers

### 6.1 Estatística Básica
- [ ] Média, mediana, desvio padrão
- [ ] Distribuições (normal, não-normal)
- [ ] Teste de hipótese (H₀, H₁, p-valor)
- [ ] Significância estatística vs significância prática
- [ ] Correlação de Pearson

### 6.2 Estatística Não-Paramétrica
- [ ] Diferença entre testes paramétricos e não-paramétricos
- [ ] Mann-Whitney U test (Wilcoxon rank-sum)
- [ ] Quando usar não-paramétrico (violação de normalidade, outliers)

### 6.3 Machine Learning Básico
- [ ] Classificação binária (duas classes)
- [ ] Treino/teste/validação
- [ ] Overfitting
- [ ] Cross-validation (k-fold, stratified)
- [ ] Métricas: acurácia, precisão, recall, F1, AUC-ROC

### 6.4 Álgebra Linear
- [ ] Vetores e matrizes
- [ ] Produto escalar
- [ ] Projeção
- [ ] Autovalores e autovetores (para PCA)

### 6.5 Lógica Fuzzy
- [ ] Diferença entre lógica crisp (0/1) e fuzzy ([0,1])
- [ ] Função de pertinência
- [ ] Operadores fuzzy (AND, OR, NOT)
- [ ] Sistemas de inferência fuzzy

---

## 7. Próximos Passos Neste Guia

Esta é a **Parte 1: Visão Geral**. Os próximos documentos detalharão:

- **GUIA_02_CARACTERISTICAS.md** - Implementação detalhada das 10 características
- **GUIA_03_ESTATISTICA.md** - Testes estatísticos passo a passo
- **GUIA_04_CLASSIFICADORES.md** - PCA, LDA, Regressão Logística explicados
- **GUIA_05_FUZZY.md** - Lógica fuzzy e funções de pertinência
- **GUIA_06_VALIDACAO.md** - Cross-validation e métricas de avaliação
- **GUIA_07_RESULTADOS.md** - Interpretação dos resultados
- **GUIA_08_PERGUNTAS_DEFESA.md** - Perguntas esperadas na defesa com respostas

---

**Próximo:** [GUIA_02_CARACTERISTICAS.md](GUIA_02_CARACTERISTICAS.md) - Implementação Detalhada das Características Estilométricas
