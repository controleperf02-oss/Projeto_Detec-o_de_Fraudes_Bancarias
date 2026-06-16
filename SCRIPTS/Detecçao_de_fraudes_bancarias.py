

# Carregamento e Análise Inicial dos Dados

import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print(df.head())

# Problema de Classificação Desbalanceada
print(df["Class"].value_counts(normalize=True))


# Engenharia de Variáveis e Padronização (Feature Engineering)

import numpy as np

df["Amount_log"] = np. log1p (df["Amount"])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df["Amount_scaled"] = scaler.fit_transform(df [["Amount"]])


# Divisão dos Dados em Treino e Teste

from sklearn.model_selection import train_test_split
X = df.drop("Class", axis=1)
y = df ["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

#Primeiro Modelo: Regressão Logística

# Logistic Regression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Avaliação do Modelo (Métricas de Correção)

from sklearn.metrics import classification_report, accuracy_score  #Está importando ferramentas prontas para medir e avaliar a qualidade das previsões da Inteligência Artificial

# 1. Mostra a porcentagem geral de acertos do modelo
print("\nAcurácia Geral do Modelo:")
print(accuracy_score(y_test, y_pred))

# 2. Mostra um relatório completo focado nas fraudes (Classe 1)
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred))


# Curvas de Análise de Performance (ROC e Precision-Recall)

# ROC Curve 
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

y_probs = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))


##################################

# Precision-Recall Curve


from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# Balanceamento de Dados (Técnicas Avançadas)

# Undersampling (reduz a classe majoritaria)
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(len(fraudes), random_state=42)

df_under = pd.concat([fraudes, normais])


# Oversampling (aumenta a classe minoritaria)
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_res, y_res = smote.fit_resample(X, y)



# Modelos de Conjunto (Random Forest e XGBoost)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
n_estimators=50,
max_depth=10,
class_weight="balanced",
n_jobs=-1,
random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(classification_report (y_test, y_pred_rf))

#################################

# Modelo Avançado - XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier(
scale_pos_weight=10, # ajuda com desbalanceamento
use_label_encoder=False,
eval_metric="logloss"
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)


# Automatização com Pipelines e Ajuste de Threshold (Limiar de Decisão)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
("scaler", StandardScaler()),
("model", LogisticRegression (max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

#################################

# threshold

threshold = 0.3

y_pred_custom = (y_probs > threshold).astype(int)

print(classification_report (y_test, y_pred_custom))



# Otimização de Hiperparâmetros (GridSearchCV) Testamos várias combinações para melhorar o modelo.

from sklearn.model_selection import GridSearchCV

param_grid = {
"max_depth": [3, 5],
"n_estimators": [50, 100]
}

grid = GridSearchCV(
XGBClassifier(eval_metric="logloss"),
param_grid,
scoring="recall",
cv=3
)

grid.fit(X_train, y_train)

print("Melhor modelo:", grid.best_params_)


# Importância de Variáveis e Explicabilidade (SHAP)

# Importância das variáveis

# Ajuda a entender quais variáveis influenciam mais o modelo.

import matplotlib.pyplot as plt

importancias = xgb.feature_importances_

plt.bar(range(len(importancias)), importancias)
plt.title("Importância das variáveis")
plt.show()

#######################################

# Explicabilidade (SHAP)

# SHAP mostra como cada variável influencia a decisão do modelo.

import shap

explainer = shap.Explainer(xgb)
shap_values = explainer(X_test[:100])

shap.plots.bar(shap_values)











