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
```
Resumo dos Resultados de Classificação por RótuloA tabela abaixo apresenta as métricas de precisão, recall, F1-score e suporte para a classe positiva (classe 1), juntamente com a acurácia geral do modelo, para cada um dos 28 rótulos avaliados.IDNome do RótuloPrecision (Cl. 1)Recall (Cl. 1)F1-Score (Cl. 1)Support (Cl. 1)Accuracy (Geral)0Hate speech0.720.390.513680.841Sexism0.610.370.462020.902Body0.790.680.73380.993Racism1.000.070.13280.984Ideology0.140.070.10280.985Homophobia0.910.640.75970.986Origin0.000.000.0080.997Religion0.000.000.0090.998Other lifestyle0.500.170.2561.009Fat people0.730.690.71350.9910Left-wing ideology0.000.000.0090.9911Ugly people0.770.770.77220.9912Black people0.670.120.21160.9913Fat women0.670.790.72280.9914Feminists0.250.160.19190.9915Gays0.700.260.38270.9916Immigrants0.000.000.0041.0017Islamists0.000.000.0071.0018Lesbians0.930.950.94591.0019Men0.730.350.48310.9920Muslims0.500.500.5021.0021Refugees0.360.380.37210.9822Trans women0.000.000.0090.9923Women0.640.360.461520.9224Transsexuals0.000.000.0061.0025Ugly women0.730.760.74210.9926Migrants0.360.360.36250.9827Homosexuals0.910.710.79820.98Observações:Desempenho Variável: O desempenho do modelo (especialmente o F1-Score para a classe 1) varia consideravelmente entre os diferentes rótulos.Baixo Suporte: Alguns rótulos (ex: Origin, Religion, Other lifestyle, Left-wing ideology, Immigrants, Islamists, Trans women, Transsexuals) têm um número muito baixo de amostras de suporte para a classe 1, o que pode tornar as métricas de precisão, recall e F1-score para essa classe menos robustas ou até mesmo nulas se o modelo não conseguir identificar corretamente nenhuma instância.Alta Acurácia Geral: A acurácia geral é consistentemente alta (muitas vezes >0.98). No entanto, isso pode ser influenciado pelo desequilíbrio de classes (muitas instâncias da classe 0 em comparação com a classe 1 para certos rótulos). O F1-score para a classe 1 é uma métrica mais indicativa do desempenho na classe minoritária.Rótulos com Melhor Desempenho na Classe 1 (F1-Score > 0.70): "Body" (0.73), "Homophobia" (0.75), "Fat people" (0.71), "Ugly people" (0.77), "Fat women" (0.72), "Lesbians" (0.94), "Ugly women" (0.74), "Homosexuals" (0.79).Rótulos com Desempenho Pobre ou Nulo na Classe 1 (F1-Score = 0.00): "Origin", "Religion", "Left-wing ideology", "Immigrants", "Islamists", "Trans women", "Transsexuals". Isto geralmente indica que o modelo não conseguiu classificar corretamente nenhuma instância da classe positiva para estes rótulos, ou que não havia instâncias suficientes para uma avaliação significativa.
```

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




