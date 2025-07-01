# Experimento: Classificação de Discurso de Ódio em Português XGBoost Multi-Label

Este repositório contém a implementação do experimento utilizando Xgboost para Multi-Label adapatado de https://gabrielziegler3.medium.com/multiclass-multilabel-classification-with-xgboost-66195e4d9f2d. 

## Descrição do Experimento
O experimento segue as etapas descritas no artigo:

1. **Carregamento dos Dados**:
   - Carrega as partições ja salvas anteriormente por ![DivisaoTreinoTeste](https://github.com/user-attachments/assets/9025a93e-c141-4d58-b593-68b27f6cbc89).
  
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





