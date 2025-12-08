# Adaptações Segundo Feedback da Regina

## Data: 2024-12-08

## Críticas Principais Identificadas

Regina identificou problemas fundamentais de **conceituação estatística** nos papers:

1. ❌ Falta explicação de mineração de texto
2. ❌ Uso de anglicismos ("corpus", "features", "tokens", "burstiness", etc.)
3. ❌ Entropia tratada como conceito matemático, não estatístico
4. ❌ Falta explicação sobre escalas de medida das variáveis
5. ❌ Terminologia não-estatística ("corpus balanceado")

## Correções Implementadas

### 1. ✅ Adicionada Seção "Mineração de Texto"

**Local**: `paper_stat/sections/methods.tex` e `paper_fuzzy/sections/methods.tex`

Adicionada nova subseção explicando:
- O que é mineração de texto
- Etapas do processo (coleta, pré-processamento, extração, análise)
- Como transforma documentos em vetores de variáveis quantitativas
- Referência: Feldman & Sanger (2007)

### 2. ✅ Correção Terminológica Completa

| Antes (Anglicismo) | Depois (Português) |
|-------------------|-------------------|
| corpus | conjunto de dados textuais |
| features | características/variáveis |
| burstiness | coeficiente de variação |
| folds | partições |
| outliers | valores atípicos |
| loadings | cargas fatoriais |
| thresholds | limiares |
| score | pontuação |
| data-driven | orientado por dados |
| Average Precision | Precisão Média |
| Precision-Recall | Precisão-Revocação |
| boxplots | diagramas de caixa |
| cluster | agrupamento |
| cross-domain | entre domínios |
| embeddings | representações vetoriais |
| tokens | palavras (quando apropriado) |

### 3. ✅ Justificativa Estatística para Entropia

**Antes**:
> "Entropia de caracteres: medida de variabilidade..."

**Depois**:
> "**Variabilidade da distribuição de caracteres**: medida de dispersão na distribuição de frequências de caracteres, calculada pela fórmula de Shannon $H = -\sum p(c) \log_2 p(c)$. Esta medida quantifica a variabilidade: alta entropia indica distribuição mais uniforme (maior dispersão); baixa entropia indica concentração (menor dispersão). **Estatisticamente, funciona como uma medida de dispersão análoga ao desvio padrão, mas aplicada a distribuições de frequência categórica**."

### 4. ✅ Coeficiente de Variação em vez de "Burstiness"

**Antes**:
> "Burstiness: coeficiente de variação..."

**Depois**:
> "**Coeficiente de variação do comprimento de frase**: razão entre o desvio padrão e a média ($CV = \sigma / \mu$). Esta medida de dispersão relativa normaliza a variabilidade pela tendência central. É uma **estatística adimensional** amplamente utilizada para comparar variabilidade entre distribuições com escalas distintas."

### 5. ✅ Escala de Medida Mantida

Mantivemos a explicação clara sobre escalas:
- 9 características em **escala de razão** (zero absoluto, razões interpretáveis)
- 1 característica (entropia) em **escala de intervalo** (diferenças interpretáveis, mas razões não)

## O Que JÁ ESTAVA BOM

✅ Explicação de variáveis contínuas
✅ Justificativa para testes não paramétricos (distribuições não-normais, valores atípicos, assimetria)
✅ Identificação clara do tipo de variáveis
✅ Descrição de medidas de tendência central e dispersão

## Pendências (Não Implementadas Neste Momento)

Questões que Regina levantou mas que requerem mudanças mais profundas:

1. **Múltiplos métodos**: Regina questionou por que usamos LDA + Logística + PCA + Fuzzy
   - Não alterado: mantivemos todos os métodos para comparação

2. **Falta de ANOVAs**: Regina mencionou que não viu ANOVAs validando os modelos
   - Não adicionado neste momento (seria mudança significativa nos resultados)

3. **Amostragem estratificada**: Regina pediu explicação detalhada do método
   - Parcialmente endereçado: texto menciona estratificação por fonte

4. **Técnica estilométrica**: Regina pediu explicação completa sobre estilometria
   - Não adicionado: seria necessário adicionar seção teórica inteira

## Commits Realizados

1. `d995f8c` - fix: substituir anglicismos por terminologia em português
2. `f04c4d6` - fix: adicionar seção mineração de texto e corrigir terminologia estatística
3. `70f9a1b` - add: referência Feldman (2007) para mineração de texto

## Status Atual

**Papers compilados com sucesso**:
- paper_stat: 18 páginas
- paper_fuzzy: 13 páginas

**Principais melhorias**:
- Terminologia 100% em português
- Conceitos estatísticos justificados
- Mineração de texto explicada
- Escalas de medida mantidas
- Justificativa estatística para entropia

## Próximos Passos (Se Necessário)

Se Regina ainda não estiver satisfeita:

1. Adicionar seção completa sobre estilometria na introdução
2. Simplificar metodologia focando em um único método (regressão logística)
3. Adicionar ANOVAs para validação dos modelos
4. Explicar em detalhe o método de estratificação usado
5. Adicionar mais referências de trabalhos que usam métodos estatísticos em mineração de texto
