O projeto de Capacitação IA e IoT faz parte de uma iniciativa liderada pela Softex em parceria com o Ministério da Ciência, Tecnologia e Inovações (MCTI), o objetivo do projeto final do Ciclo 2 é aplicar o conhecimento adquirido ao longo do ciclo, especialmente em relação aos modelos de aprendizado de máquina.

Etapas do projeto:

Obter um Dataset: Os participantes devem escolher um conjunto de dados de classificação ou regressão de fontes diversas, como SkLearn, Seaborn, UCI, Kaggle ou qualquer outra fonte relevante.

Análise de Dados com Pandas: Os dados devem ser explorados e preparados usando o Pandas. Isso inclui verificar a presença de valores nulos, verificar a correspondência entre as features, converter labels categóricos para números e normalizar as features para a escala 0..1.

Execução de Experimentos com MLflow:

3.1) Escolher 3 algoritmos de aprendizado de máquina com 3 variações de parâmetros cada, totalizando 9 variações.

3.2) Testar Bagging e RandomForest com parâmetros padrão ou personalizados.

3.3) Executar experimentos com GradientBoosting, XGBoost e LightGBM, com parâmetros padrão ou personalizados.

3.4) Testar métodos de seleção dinâmica com os modelos gerados com Bagging e RandomForest.

3.5) Calcular métricas como Acurácia, Precision, Recall, Sensibilidade, AUC (classificação) ou MSE, RMSE, MAPE (regressão) usando validação cruzada de 10 folds.
3.6) Utilizar 2 métodos de seleção de features para todos os experimentos.
3.7) Registrar todas as execuções no MLflow Tracking.
3.8) Registrar todos os parâmetros e métricas dos modelos usando autolog() no MLflow.
3.9) Registrar os 3 modelos com melhor desempenho com base na acurácia.
3.10) Escrever um código cliente para carregar os modelos registrados e exibir suas informações, incluindo descrições.
