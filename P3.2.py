from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import mlflow
import numpy as np

# Nome do experimento no MLflow
exp_mlflow_ciclo2 = "exp_projeto_ciclo_2"

# Iniciar o tracking do MLflow
mlflow.set_tracking_uri("https://ef00-35-231-71-79.ngrok-free.app/")
mlflow.set_experiment(exp_mlflow_ciclo2)

# Carregar o conjunto de dados
data = load_breast_cancer()
X, y = data.data, data.target

# Definir os parâmetros padrão e customizados para RandomForest
random_forest_default_params = {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
random_forest_custom_params = {"n_estimators": 50, "max_depth": 5, "min_samples_split": 5}

# Definir os parâmetros padrão e customizados para Bagging
bagging_default_params = {"n_estimators": 100}
bagging_custom_params = {"n_estimators": 50}

# Função que vai rodar experimentos para um algoritmo específico
def run_experiment(model, params, algorithm_name):
    for i, param_set in enumerate(params):
        # Calcular as métricas utilizando validação cruzada de 10 folds
        scores = cross_val_score(model(**param_set), X, y, cv=10)
        # Calcular a média das métricas
        mean_accuracy = scores.mean()
        print(f"{algorithm_name} com parâmetros {param_set} - Variação {i+1}: Acurácia Média = {mean_accuracy:.4f}")
        # Log das métricas no MLflow
        with mlflow.start_run(run_name=f"{algorithm_name}_var{i+1}", nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("accuracy", mean_accuracy)

            # Treinar o classificador com todos os dados e log do modelo treinado
            model_instance = model(**param_set)
            model_instance.fit(X, y)
            mlflow.sklearn.log_model(model_instance, "model")

# Rodar experimentos para o algoritmo RandomForest com parâmetros padrão e customizados
print("\nRandomForest com parâmetros padrão:")
run_experiment(RandomForestClassifier, [random_forest_default_params], "RandomForest_Default")
print("\nRandomForest com parâmetros customizados:")
run_experiment(RandomForestClassifier, [random_forest_custom_params], "RandomForest_Custom")

# Rodar experimentos para o algoritmo Bagging com parâmetros padrão e customizados
print("\nBagging com parâmetros padrão:")
run_experiment(BaggingClassifier, [bagging_default_params], "Bagging_Default")
print("\nBagging com parâmetros customizados:")
run_experiment(BaggingClassifier, [bagging_custom_params], "Bagging_Custom")