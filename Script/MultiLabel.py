import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.model_selection import iterative_train_test_split
from nltk.corpus import stopwords
import logging

nltk.download('stopwords')

# Configurações de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

def load_and_prepare_data(file_path):
    try:
        logging.info("Carregando os dados...")
        data = pd.read_csv(file_path)
        data['text'] = data['text'].apply(clean_text)
        X = data['text']
        y = data.drop(columns=['text'])
        return X, y
    except FileNotFoundError:
        logging.error(f"Arquivo {file_path} não encontrado.")
        return None, None

def filter_labels(y, min_count=10):
    label_counts = y.sum(axis=0)
    valid_labels = label_counts[label_counts >= min_count].index
    return y[valid_labels]

def train_and_evaluate(X_train, y_train, X_test, y_test):
    params = {
        'max_depth': 6,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    models = {}
    predictions = {}

    for label_idx in range(y_train.shape[1]):
        logging.info(f"Treinando modelo para o rótulo: {label_idx}")
        dtrain_label = xgb.DMatrix(data=X_train, label=y_train[:, label_idx])
        model = xgb.train(params, dtrain_label, num_boost_round=100)
        models[label_idx] = model

        logging.info(f"Fazendo previsões para o rótulo: {label_idx}")
        dtest_label = xgb.DMatrix(data=X_test)
        predictions[label_idx] = model.predict(dtest_label)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv("predictions.csv", index=False)
    logging.info("Previsões salvas em predictions.csv")

    for label_idx in range(y_test.shape[1]):
        logging.info(f"Avaliando o rótulo: {label_idx}")
        print(classification_report(
            y_test[:, label_idx],
            (predictions[label_idx] >= 0.5).astype(int),
            zero_division=0
        ))

def gerar():
    X, y = load_and_prepare_data('2019-05-28_portuguese_hate_speech_hierarchical_classification.csv')
    if X is None or y is None:
        return

    y = filter_labels(y)
    logging.info(f"Rótulos mantidos: {list(y.columns)}")

    portuguese_stopwords = stopwords.words('portuguese')
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=portuguese_stopwords)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, y_train, X_test, y_test = iterative_train_test_split(X_tfidf, y.values, test_size=0.3)
    logging.info(f"Formatos: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    train_and_evaluate(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    gerar()



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