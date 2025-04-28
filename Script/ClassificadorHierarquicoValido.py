import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Carrega a base de dados
data = pd.read_csv('Data/2019-05-28_portuguese_hate_speech_hierarchical_classification_reduzido.csv', on_bad_lines='skip')

# Divide os dados em texto e rótulos
X = data['text']  # Coluna com o texto
y = data.drop(columns=['text'])  # Todas as colunas exceto o texto

# Verifica os tipos de dados em y
print("Tipos de dados em y antes da conversão:")
print(y.dtypes)

# Converte todas as colunas de y para valores numéricos
y = y.apply(pd.to_numeric, errors='coerce')

# Substitui valores NaN por 0
y = y.fillna(0)

# Substitui valores fora do intervalo [0, 1] por 0
y[y > 1] = 0

# Verifica os valores únicos em y
print("Valores únicos em y após a conversão:", np.unique(y.values))

# Converte os rótulos para inteiros e depois para uma matriz NumPy
y = y.astype(int).values

# Vetorização do texto usando TF-IDF
portuguese_stopwords = stopwords.words('portuguese')
vectorizer = TfidfVectorizer(max_features=5000, stop_words=portuguese_stopwords)
X_tfidf = vectorizer.fit_transform(X)

# Stratificação hierárquica usando scikit-multilearn
X_train, y_train, X_test, y_test = iterative_train_test_split(
    X_tfidf, y, test_size=0.3
)

# Verifica a distribuição das classes no conjunto de treino e teste
print("Distribuição das classes no conjunto de treino:")
print(y_train.sum(axis=0))

print("Distribuição das classes no conjunto de teste:")
print(y_test.sum(axis=0))

# Verifica os formatos após a divisão
print("Formatos:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# Configura os parâmetros do XGBoost
params = {
    'max_depth': 6,
    'objective': 'binary:logistic',  # Classificação binária
    'eval_metric': 'logloss',
}

# Treina um modelo XGBoost para cada rótulo
models = {}
for label_idx in range(y_train.shape[1]):
    print(f"Treinando modelo para o rótulo: {label_idx}")
    dtrain_label = xgb.DMatrix(data=X_train, label=y_train[:, label_idx])
    model = xgb.train(params, dtrain_label, num_boost_round=100)
    models[label_idx] = model

# Faz previsões para cada rótulo
predictions = {}
for label_idx, model in models.items():
    print(f"Fazendo previsões para o rótulo: {label_idx}")
    dtest_label = xgb.DMatrix(data=X_test)
    predictions[label_idx] = model.predict(dtest_label)

# Converte as previsões para um DataFrame
predictions_df = pd.DataFrame(predictions)
print(predictions_df.head())

# Avalia o modelo com zero_division=0
for label_idx in range(y_test.shape[1]):
    print(f"Avaliando o rótulo: {label_idx}")
    print(classification_report(y_test[:, label_idx], (predictions[label_idx] >= 0.5).astype(int), zero_division=0))

print("terminou")
'''    
classes = [f"Classe {i}" for i in range(len(y_train.sum(axis=0)))]
train_distribution = y_train.sum(axis=0)
test_distribution = y_test.sum(axis=0)

# Configura o tamanho da figura
plt.figure(figsize=(12, 6))

# Plotagem da distribuição no conjunto de treino
plt.bar(np.arange(len(classes)) - 0.2, train_distribution, width=0.4, label="Treino", color="blue")

# Plotagem da distribuição no conjunto de teste
plt.bar(np.arange(len(classes)) + 0.2, test_distribution, width=0.4, label="Teste", color="orange")

# Configurações do gráfico
plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
plt.xlabel("Classes")
plt.ylabel("Número de Amostras")
plt.title("Distribuição das Classes no Conjunto de Treino e Teste")
plt.legend()
plt.tight_layout()

# Exibe o gráfico
plt.show()

# Calcula a matriz de confusão para cada rótulo
confusion_matrices = multilabel_confusion_matrix(y_test, (predictions_df >= 0.5).astype(int))

# Plotagem da matriz de confusão para cada rótulo
for label_idx, matrix in enumerate(confusion_matrices):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"])
    plt.title(f"Matriz de Confusão - Classe {label_idx}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.show()
'''