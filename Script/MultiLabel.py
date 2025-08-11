import pandas as pd
import numpy as np
import re
import os
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from nltk.corpus import stopwords
import xgboost as xgb

logging.basicConfig(level=logging.INFO)

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

def gerar_particoes_multilabel(X_tfidf, y, n_splits=10, caminho='particoes.pkl'):
    logging.info(f"📁 Gerando {n_splits} partições multilabel para validação cruzada...")

    os.makedirs(os.path.dirname(caminho), exist_ok=True)

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []

    for train_idx, test_idx in mskf.split(X_tfidf, y):
        folds.append((train_idx, test_idx))

    with open(caminho, 'wb') as f:
        pickle.dump(folds, f)

    logging.info(f"✅ Partições salvas em {os.path.abspath(caminho)}")

def train_cross_validation(X, y, folds):
    params = {
        'max_depth': 6,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    all_predictions = np.zeros(y.shape)

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        logging.info(f"🔁 Fold {fold_idx + 1}/{len(folds)}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for label_idx in range(y.shape[1]):
            logging.info(f"Treinando rótulo {label_idx} no Fold {fold_idx + 1}")
            dtrain = xgb.DMatrix(X_train, label=y_train[:, label_idx])
            dtest = xgb.DMatrix(X_test)

            model = xgb.train(params, dtrain, num_boost_round=100)
            preds = model.predict(dtest)

            all_predictions[test_idx, label_idx] = preds

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv("predictions_crossval.csv", index=False)
    logging.info("📄 Previsões salvas em predictions_crossval.csv")
    # Após gerar predictions_df    

    # Avaliação geral
    for label_idx in range(y.shape[1]):
        logging.info(f"Avaliando o rótulo: {label_idx}")
        print(classification_report(
            y[:, label_idx],
            (all_predictions[:, label_idx] >= 0.5).astype(int),
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

    if not os.path.exists("particoes.pkl"):
        gerar_particoes_multilabel(X_tfidf, y.values, n_splits=5, caminho="particoes.pkl")

    with open("particoes.pkl", "rb") as f:
        folds = pickle.load(f)

    train_cross_validation(X_tfidf, y.values, folds)

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