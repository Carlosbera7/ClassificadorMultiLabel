# Experimento: Classificação de Discurso de Ódio em Português XGBoost Multi-Label

Este repositório contém a implementação do experimento utilizando Xgboost para Multi-Label adapatado de https://gabrielziegler3.medium.com/multiclass-multilabel-classification-with-xgboost-66195e4d9f2d. 

## Descrição do Experimento
O experimento segue as etapas descritas no artigo:

1. **Carregamento dos Dados**:
   - O arquivo CSV 2019-05-28_portuguese_hate_speech_hierarchical_classification_reduzido.csv é carregado.
   - A coluna text é separada como as features (X), e as demais colunas são tratadas como rótulos (y).

2. **Pré-processamento dos Rótulos**:
     - Os rótulos (y) são convertidos para valores numéricos
     - Valores inválidos ou fora do intervalo [0, 1] são substituídos por 0.
     - Valores NaN são preenchidos com 0.
     - Os rótulos são convertidos para inteiros e transformados em uma matriz NumPy.   

3. **Vetorização do Texto**:
   - O texto (X) é vetorizado usando TF-IDF com um limite de 5000 features.
   - Stopwords em português são removidas utilizando a biblioteca NLTK.
      
4. **Divisão dos Dados**:
   - Os dados são divididos em conjuntos de treino e teste utilizando stratificação hierárquica com a função iterative_train_test_split da biblioteca scikit-multilearn.
   - A distribuição das classes nos conjuntos de treino e teste é verificada.
  
5. **Treinamento do Modelo**:
   - Um modelo XGBoost é treinado para cada rótulo (coluna de y).
   - O modelo utiliza a função de objetivo binary:logistic para classificação binária.
     
## Implementação
O experimento foi implementado em Python 3.6 utilizando as bibliotecas:
- pandas
- NLTK
- Scikit-learn
- XGBoost

## Divisão
![Divisao](https://github.com/user-attachments/assets/7da2dc03-7fc2-4680-8d21-094c31f174a9)

O script principal executa as seguintes etapas:
1. Carregamento das partições salvas.
2. Tokenização e padding das sequências de texto.
3. Carregamento dos embeddings GloVe.
4. Construção e treinamento do modelo LSTM.
5. Extração das representações intermediárias.
6. Treinamento e avaliação do XGBoost.
7. Busca de hiperparâmetros com validação cruzada.

## Estrutura do Repositório
- [`Scripts/ClassificadorHierarquicoValido.py`](https://github.com/Carlosbera7/ClassificadorMultiLabel/blob/main/Script/ClassificadorHierarquicoValido.py): Script principal para executar o experimento.
- [`Data/`](https://github.com/Carlosbera7/ClassificadorMultiLabel/tree/main/Data): Pasta contendo o conjunto de dados e o Embeddings GloVe pré-treinados (necessário para execução).
- [`Execução`](https://musical-space-yodel-9rpvjvw9qr39vw4.github.dev/): O código pode ser executado diretamente no ambiente virtual.

## Resultados

| Rótulo | Precisão (0) | Recall (0) | F1-Score (0) | Suporte (0) | Precisão (1) | Recall (1) | F1-Score (1) | Suporte (1) | Acurácia | Macro Avg. (F1) | Weighted Avg. (F1) |
| ------ | ------------ | ---------- | ------------ | ----------- | ------------ | ---------- | ------------ | ----------- | -------- | --------------- | ------------------ |
| Hate.speech      | 0.85         | 0.96       | 0.90         | 1333        | 0.72         | 0.39       | 0.51         | 368         | 0.84     | 0.71            | 0.82               |
| Sexism      | 0.92         | 0.97       | 0.94         | 1499        | 0.61         | 0.37       | 0.46         | 202         | 0.90     | 0.70            | 0.89               |
| Body      | 0.99         | 1.00       | 0.99         | 1663        | 0.79         | 0.68       | 0.73         | 38          | 0.99     | 0.86            | 0.99               |
| Racism      | 0.98         | 1.00       | 0.99         | 1673        | 1.00         | 0.07       | 0.13         | 28          | 0.98     | 0.56            | 0.98               |
| Ideology      | 0.98         | 0.99       | 0.99         | 1673        | 0.14         | 0.07       | 0.10         | 28          | 0.98     | 0.54            | 0.97               |
| Homophobia      | 0.98         | 1.00       | 0.99         | 1604        | 0.91         | 0.64       | 0.75         | 97          | 0.98     | 0.87            | 0.97               |
| Origin      | 1.00         | 1.00       | 1.00         | 1693        | 0.00         | 0.00       | 0.00         | 8           | 0.99     | 0.50            | 0.99               |
| Religion      | 0.99         | 1.00       | 1.00         | 1692        | 0.00         | 0.00       | 0.00         | 9           | 0.99     | 0.50            | 0.99               |
| OtherLifestyle      | 1.00         | 1.00       | 1.00         | 1695        | 0.50         | 0.17       | 0.25         | 6           | 1.00     | 0.62            | 1.00               |
| Fat.people      | 0.99         | 0.99       | 0.99         | 1666        | 0.73         | 0.69       | 0.71         | 35          | 0.99     | 0.85            | 0.99               |
| Left.wing.ideology     | 0.99         | 1.00       | 1.00         | 1692        | 0.00         | 0.00       | 0.00         | 9           | 0.99     | 0.50            | 0.99               |
| Ugly.people     | 1.00         | 1.00       | 1.00         | 1679        | 0.77         | 0.77       | 0.77         | 22          | 0.99     | 0.88            | 0.99               |
| Black.people     | 0.99         | 1.00       | 1.00         | 1685        | 0.67         | 0.12       | 0.21         | 16          | 0.99     | 0.60            | 0.99               |
| Fat.women     | 1.00         | 0.99       | 0.99         | 1673        | 0.67         | 0.79       | 0.72         | 28          | 0.99     | 0.86            | 0.99               |
| Feminists     | 0.99         | 0.99       | 0.99         | 1682        | 0.25         | 0.16       | 0.19         | 19          | 0.99     | 0.59            | 0.98               |
| Gays     | 0.99         | 1.00       | 0.99         | 1674        | 0.70         | 0.26       | 0.38         | 27          | 0.99     | 0.69            | 0.98               |
| Immigrants     | 1.00         | 1.00       | 1.00         | 1697        | 0.00         | 0.00       | 0.00         | 4           | 1.00     | 0.50            | 1.00               |
| Islamists     | 1.00         | 1.00       | 1.00         | 1694        | 0.00         | 0.00       | 0.00         | 7           | 1.00     | 0.50            | 0.99               |
| Lesbians     | 1.00         | 1.00       | 1.00         | 1642        | 0.93         | 0.95       | 0.94         | 59          | 1.00     | 0.97            | 1.00               |
| Men     | 0.99         | 1.00       | 0.99         | 1670        | 0.73         | 0.35       | 0.48         | 31          | 0.99     | 0.74            | 0.98               |
| Muslims     | 1.00         | 1.00       | 1.00         | 1699        | 0.50         | 0.50       | 0.50         | 2           | 1.00     | 0.75            | 1.00               |
| Refugees     | 0.99         | 0.99       | 0.99         | 1680        | 0.36         | 0.38       | 0.37         | 21          | 0.98     | 0.68            | 0.98               |
| Trans.women     | 0.99         | 1.00       | 1.00         | 1692        | 0.00         | 0.00       | 0.00         | 9           | 0.99     | 0.50            | 0.99               |
| Women     | 0.94         | 0.98       | 0.96         | 1549        | 0.64         | 0.36       | 0.46         | 152         | 0.92     | 0.71            | 0.92               |
| Transexuals     | 1.00         | 1.00       | 1.00         | 1695        | 0.00         | 0.00       | 0.00         | 6           | 1.00     | 0.50            | 0.99               |
| Ugly.women     | 1.00         | 1.00       | 1.00         | 1680        | 0.73         | 0.76       | 0.74         | 21          | 0.99     | 0.87            | 0.99               |
| Migrants     | 0.99         | 0.99       | 0.99         | 1676        | 0.36         | 0.36       | 0.36         | 25          | 0.98     | 0.68            | 0.98               |
| Homossexuals     | 0.99         | 1.00       | 0.99         | 1619        | 0.91         | 0.71       | 0.79         | 82          | 0.98     | 0.89            | 0.98               |


Os resultados incluem:

Exemplo das 5 primeiras linhas das Predições : 

         0         1         2         3         4   ...        23        24        25        26        27
0  0.822482  0.011491  0.000115  0.003364  0.015925  ...  0.004265  0.000120  0.001934  0.061836  0.005956
1  0.244313  0.009504  0.000059  0.000159  0.000568  ...  0.009568  0.000025  0.000069  0.000182  0.001240
2  0.363881  0.003745  0.000245  0.000483  0.001907  ...  0.001373  0.000053  0.000473  0.000115  0.003684
3  0.048330  0.006886  0.000459  0.000477  0.000181  ...  0.003678  0.000150  0.000025  0.000037  0.000387
4  0.337050  0.473267  0.000016  0.006089  0.003918  ...  0.024235  0.000120  0.000419  0.000021  0.006913
```

Exemplo de saída dos 5 primeiros rótulos :
```

Avaliando o rótulo: 0
              precision    recall  f1-score   support

           0       0.90      0.96      0.93      1473
           1       0.59      0.37      0.46       243

    accuracy                           0.87      1716
   macro avg       0.75      0.67      0.69      1716
weighted avg       0.86      0.87      0.86      1716

```

![ConfuseM](https://github.com/user-attachments/assets/149bf533-2049-475f-9875-01a19dbb2044)

```
Avaliando o rótulo: 1
              precision    recall  f1-score   support

           0       0.94      0.97      0.96      1587
           1       0.45      0.30      0.36       129

    accuracy                           0.92      1716
   macro avg       0.70      0.64      0.66      1716
weighted avg       0.91      0.92      0.91      1716
```
![ConfuseM1](https://github.com/user-attachments/assets/3339dc65-5205-4552-b97f-0bcfb5d8f435)

```
Avaliando o rótulo: 2
              precision    recall  f1-score   support

           0       1.00      0.99      0.99      1692
           1       0.62      0.67      0.64        24

    accuracy                           0.99      1716
   macro avg       0.81      0.83      0.82      1716
weighted avg       0.99      0.99      0.99      1716
```
![ConfuseM2](https://github.com/user-attachments/assets/fddb34bb-ab25-44a9-b41e-3faff04f0e16)

```
Avaliando o rótulo: 3
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      1695
           1       0.50      0.10      0.16        21

    accuracy                           0.99      1716
   macro avg       0.74      0.55      0.58      1716
weighted avg       0.98      0.99      0.98      1716
```
![ConfuseM3](https://github.com/user-attachments/assets/1bade5fb-688b-43ce-b1bc-af7d6f977cbd)

```
Avaliando o rótulo: 4
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1696
           1       0.25      0.15      0.19        20

    accuracy                           0.98      1716
   macro avg       0.62      0.57      0.59      1716
weighted avg       0.98      0.98      0.98      1716
```
![ConfuseM4](https://github.com/user-attachments/assets/741236e1-3a13-4f38-afd7-61d1df0483dc)




