from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
import numpy as np

# Definir experimento e conectar ao MLflow
exp_mlflow_ciclo2 = "exp_projeto_ciclo_2"
mlflow.set_tracking_uri("https://ef00-35-231-71-79.ngrok-free.app/")
mlflow.set_experiment(exp_mlflow_ciclo2)

# Carregar o conjunto de dados
data = load_breast_cancer()
X, y = data.data, data.target

# Definir as variações de parâmetros para cada algoritmo
logistic_regression_params = [
    {"penalty": "l1", "C": 1.0, "solver": "liblinear"},
    {"penalty": "l2", "C": 0.5, "solver": "liblinear"},
    {"penalty": "l2", "C": 0.1, "solver": "liblinear"}
]

knn_params = [
    {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
    {"n_neighbors": 10, "weights": "distance", "algorithm": "auto"},
    {"n_neighbors": 15, "weights": "uniform", "algorithm": "ball_tree"}
]

decision_tree_params = [
    {"criterion": "gini", "max_depth": None, "min_samples_split": 2},
    {"criterion": "entropy", "max_depth": 5, "min_samples_split": 5},
    {"criterion": "gini", "max_depth": 10, "min_samples_split": 10}
]

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
        print(f"{algorithm_name} - Variação {i+1}: Acurácia Média = {mean_accuracy:.4f}")
        # Log das métricas no MLflow
        with mlflow.start_run(run_name=f"{algorithm_name}_var{i+1}", nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("accuracy", mean_accuracy)

            # Treinar o classificador com todos os dados e log do modelo treinado
            model_instance = model(**param_set)
            model_instance.fit(X, y)
            mlflow.sklearn.log_model(model_instance, "model")

# Rodar experimentos para o algoritmo Regressão Logística
print("Regressão Logística:")
run_experiment(LogisticRegression, logistic_regression_params, "LogisticRegression")

# Rodar experimentos para o algoritmo KNN
print("\nKNN:")
run_experiment(KNeighborsClassifier, knn_params, "KNN")

# Rodar experimentos para o algoritmo Árvore de Decisão
print("\nÁrvore de Decisão:")
run_experiment(DecisionTreeClassifier, decision_tree_params, "DecisionTree")