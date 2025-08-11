# Experimento: Classificação de Discurso de Ódio em Português XGBoost Multi-Label

Este repositório contém a implementação do experimento utilizando Xgboost para Multi-Label adapatado de https://gabrielziegler3.medium.com/multiclass-multilabel-classification-with-xgboost-66195e4d9f2d. 

## Descrição do Experimento
O experimento segue as etapas descritas no artigo:

1. **Carregamento dos Dados**:
   - Carrega as partições ja salvas anteriormente por https://github.com/Carlosbera7/SalvarParticoesMultiLabel.
  
2. **Treinamento do Modelo**:
   - Um modelo XGBoost é treinado para cada rótulo (coluna de y).
   - O modelo utiliza a função de objetivo binary:logistic para classificação binária.
   - O modelo treinado do XgBoost é salvo em Data.
     
## Implementação
O experimento foi implementado em Python 3.6 utilizando as bibliotecas:
- pandas
- NLTK
- Scikit-learn
- XGBoost

## Divisão

Divisão das Partições :
![DivisaoTreinoTeste](https://github.com/user-attachments/assets/9025a93e-c141-4d58-b593-68b27f6cbc89)


Detalhamento da partição de Treino :
![image](https://github.com/user-attachments/assets/dd2d8d46-41b8-448e-8516-671de5cadc13)

Detalhamento da partição de Teste :
![image](https://github.com/user-attachments/assets/5b141df6-de9d-4437-a99c-0a30a8401d76)


## Estrutura do Repositório
- [`Scripts/MultiLabel.py`](https://github.com/Carlosbera7/ClassificadorMultiLabel/blob/main/Script/MultiLabel.py): Script principal para executar o experimento.
- [`Data/`](https://github.com/Carlosbera7/ClassificadorMultiLabel/tree/main/Data): Pasta contendo o conjunto de dados e o modelo treinado do XGBoost para cada rótulo. 

## Resultados


| Rótulo | Prec. (0) | Recall (0) | F1 (0) | Prec. (1) | Recall (1) | F1 (1) | Acurácia |
|--------|-----------|------------|--------|-----------|------------|--------|----------|
| Hate.speech      | 0.88      | 0.97       | 0.92   | 0.81      | 0.53       | 0.64   | 0.87     |
| Sexism      | 0.94      | 0.97       | 0.96   | 0.75      | 0.56       | 0.64   | 0.93     |
| Body      | 1.00      | 1.00       | 1.00   | 0.88      | 0.84       | 0.86   | 0.99     |
| Racism      | 0.99      | 1.00       | 0.99   | 0.46      | 0.17       | 0.25   | 0.98     |
| Ideology      | 0.99      | 1.00       | 0.99   | 0.53      | 0.21       | 0.30   | 0.98     |
| Homophobia      | 0.99      | 1.00       | 0.99   | 0.94      | 0.78       | 0.85   | 0.98     |
| Origin      | 1.00      | 1.00       | 1.00   | 0.00      | 0.00       | 0.00   | 0.99     |
| Religion      | 1.00      | 1.00       | 1.00   | 0.15      | 0.07       | 0.09   | 0.99     |
| OtherLifestyle      | 1.00      | 1.00       | 1.00   | 0.50      | 0.15       | 0.23   | 1.00     |
| Fat.people      | 1.00      | 1.00       | 1.00   | 0.87      | 0.84       | 0.86   | 0.99     |
| Left.wing.ideology     | 1.00      | 1.00       | 1.00   | 0.00      | 0.00       | 0.00   | 1.00     |
| Ugly.people     | 1.00      | 1.00       | 1.00   | 0.84      | 0.85       | 0.84   | 0.99     |
| Black.people     | 0.99      | 1.00       | 1.00   | 0.46      | 0.25       | 0.33   | 0.99     |
| Fat.women     | 1.00      | 1.00       | 1.00   | 0.85      | 0.88       | 0.86   | 0.99     |
| Feminists     | 0.99      | 1.00       | 0.99   | 0.47      | 0.26       | 0.34   | 0.99     |
| Gays     | 0.99      | 1.00       | 1.00   | 0.73      | 0.29       | 0.41   | 0.99     |
| Immigrants     | 1.00      | 1.00       | 1.00   | 0.00      | 0.00       | 0.00   | 1.00     |
| Islamists     | 1.00      | 1.00       | 1.00   | 0.09      | 0.06       | 0.07   | 1.00     |
| Lesbians     | 1.00      | 1.00       | 1.00   | 0.96      | 0.96       | 0.96   | 1.00     |
| Men     | 0.99      | 1.00       | 0.99   | 0.47      | 0.30       | 0.37   | 0.99     |
| Muslims     | 1.00      | 1.00       | 1.00   | 0.29      | 0.18       | 0.22   | 1.00     |
| Refugees     | 0.99      | 1.00       | 0.99   | 0.37      | 0.21       | 0.27   | 0.99     |
| Trans.women     | 1.00      | 1.00       | 1.00   | 0.33      | 0.04       | 0.07   | 1.00     |
| Women     | 0.96      | 0.98       | 0.97   | 0.77      | 0.57       | 0.66   | 0.94     |
| Transexuals     | 1.00      | 1.00       | 1.00   | 0.00      | 0.00       | 0.00   | 1.00     |
| Ugly.women     | 1.00      | 1.00       | 1.00   | 0.83      | 0.85       | 0.84   | 0.99     |
| Migrants     | 0.99      | 1.00       | 0.99   | 0.43      | 0.22       | 0.29   | 0.98     |
| Homossexuals     | 0.99      | 1.00       | 0.99   | 0.95      | 0.84       | 0.89   | 0.99     |






