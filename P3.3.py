from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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

# Função para treinar e avaliar o classificador
def train_and_evaluate_classifier(model, X, y):
    # Treinar o classificador
    model.fit(X, y)

    # Fazer previsões no conjunto de dados
    y_pred = model.predict(X)

    # Calcular a acurácia
    accuracy = accuracy_score(y, y_pred)
    return accuracy

# Função que vai rodar experimentos para um algoritmo específico
def run_experiment(model, params, algorithm_name):
    for i, param_set in enumerate(params):
        # Calcular as métricas utilizando validação cruzada de 10 folds
        scores = cross_val_score(model(**param_set), X, y, cv=10)
        # Calcular a média das métricas
        mean_accuracy = np.mean(scores)
        print(f"{algorithm_name} com parâmetros {param_set} - Variação {i+1}: Acurácia Média = {mean_accuracy:.4f}")
        # Log das métricas no MLflow
        with mlflow.start_run(run_name=f"{algorithm_name}_var{i+1}", nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("accuracy", mean_accuracy)

            # Treinar o classificador com todos os dados e log do modelo treinado
            model_instance = model(**param_set)
            model_instance.fit(X, y)
            mlflow.sklearn.log_model(model_instance, "model")

# Rodar experimentos para o algoritmo GradientBoosting com parâmetros padrão e customizados
gradient_boosting_default_params = {}  # Definir os parâmetros padrão para GradientBoosting
gradient_boosting_custom_params = {}   # Definir os parâmetros customizados para GradientBoosting
print("\nGradientBoosting com parâmetros padrão:")
run_experiment(GradientBoostingClassifier, [gradient_boosting_default_params], "GradientBoosting_Default")
print("\nGradientBoosting com parâmetros customizados:")
run_experiment(GradientBoostingClassifier, [gradient_boosting_custom_params], "GradientBoosting_Custom")

# Rodar experimentos para o algoritmo XGBoost com parâmetros padrão e customizados
xgboost_default_params = {}  # Definir os parâmetros padrão para XGBoost
xgboost_custom_params = {}   # Definir os parâmetros customizados para XGBoost
print("\nXGBoost com parâmetros padrão:")
run_experiment(XGBClassifier, [xgboost_default_params], "XGBoost_Default")
print("\nXGBoost com parâmetros customizados:")
run_experiment(XGBClassifier, [xgboost_custom_params], "XGBoost_Custom")

# Rodar experimentos para o algoritmo LightGBM com parâmetros padrão e customizados
lightgbm_default_params = {}  # Definir os parâmetros padrão para LightGBM
lightgbm_custom_params = {}   # Definir os parâmetros customizados para LightGBM
print("\nLightGBM com parâmetros padrão:")
run_experiment(LGBMClassifier, [lightgbm_default_params], "LightGBM_Default")
print("\nLightGBM com parâmetros customizados:")
run_experiment(LGBMClassifier, [lightgbm_custom_params], "LightGBM_Custom")