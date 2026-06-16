# 💳 Projeto: Detecção de Fraudes em Cartões de Crédito com Machine Learning

Este projeto aborda um dos problemas mais desafiadores e importantes nas instituições financeiras: a detecção de fraudes em transações bancárias. O objetivo é construir um pipeline de Machine Learning capaz de identificar transações fraudulentas de alta precisão, lidando com o extremo desbalanceamento dos dados.

---

## 🛠️ Arquitetura e Explicação Passo a Passo do Código
O desenvolvimento do projeto foi dividido em etapas lógicas, cobrindo desde a importação de dados até a explicabilidade do modelo final.

### 1. 📊 Carregamento e Análise Inicial dos Dados
O projeto inicia com o carregamento do dataset e a análise da distribuição da variável alvo (`Class`), onde `0` representa transações legítimas e `1` representa fraudes.

* **Para que serve:** Carrega a base de dados diretamente de um repositório na nuvem e exibe as 5 primeiras linhas para entender a estrutura.
* **Por que foi utilizado:** Para diagnosticar o nível de desbalanceamento. Neste dataset, menos de 0.2% das transações são fraudes. Se ignorarmos isso, um modelo burro que diz que "toda transação é legítima" teria 99.8% de acurácia, mas não detectaria nenhuma fraude.

```python
import pandas as pd

url = "[https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)"
df = pd.read_csv(url)

print(df.head())

# Problema de Classificação Desbalanceada
print(df["Class"].value_counts(normalize=True))

### 2. ⚙️ Engenharia de Variáveis e Padronização (Feature Engineering)
Modelos lineares e redes neurais funcionam muito melhor quando os dados numéricos estão na mesma escala ou distribuídos de forma normal.

* **Para que serve:** O `np.log1p` aplica uma transformação logarítmica na coluna de valores (`Amount`) para reduzir o impacto de valores extremamente altos (outliers). O `StandardScaler` reescala os dados para que a média seja 0 e o desvio padrão seja 1.
* **Por que foi utilizado:** Evita que os valores das transações (que variam de centavos a milhares de reais) dominem os algoritmos e distorçam o aprendizado em relação às outras variáveis (que já vêm transformadas por PCA).
