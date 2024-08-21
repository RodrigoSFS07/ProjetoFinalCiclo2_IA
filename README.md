Projeto de Capacitação IA e IoT

O projeto de Capacitação IA e IoT faz parte de uma iniciativa liderada pela Softex em parceria com o Ministério da Ciência, Tecnologia e Inovações (MCTI). O objetivo é aplicar o conhecimento adquirido no Ciclo 2, focando em modelos de aprendizado de máquina.

<div align="center">
    <img src="https://github.com/user-attachments/assets/9827d17c-620d-4206-934b-a1149abb75da" alt="Projeto de Capacitação IA e IoT" />
</div>

Etapas do Projeto

1. **Obtenção de Dataset**: Escolher um conjunto de dados de classificação ou regressão de fontes como SkLearn, Seaborn, UCI, Kaggle, entre outros.

2. **Análise de Dados com Pandas**:
   - Explorar e preparar os dados: verificar valores nulos, converter labels categóricos para numéricos, e normalizar features.

3. **Execução de Experimentos com MLflow**:
   - **Escolha e Teste de Algoritmos**:
     - Testar 3 algoritmos de aprendizado de máquina com 3 variações de parâmetros cada.
     - Testar Bagging e RandomForest com parâmetros padrão ou personalizados.
     - Realizar experimentos com GradientBoosting, XGBoost e LightGBM.
     - Utilizar métodos de seleção dinâmica com Bagging e RandomForest.
   - **Avaliação e Registro**:
     - Calcular métricas como Acurácia, Precision, Recall, Sensibilidade, AUC (classificação) ou MSE, RMSE, MAPE (regressão) usando validação cruzada de 10 folds.
     - Utilizar 2 métodos de seleção de features.
     - Registrar todas as execuções e parâmetros no MLflow Tracking com `autolog()`.
     - Registrar os 3 melhores modelos com base na acurácia.
     - Desenvolver um código cliente para carregar e exibir informações dos modelos registrados.

Este resumo proporciona uma visão geral das etapas principais e objetivos do projeto de forma concisa e clara.
