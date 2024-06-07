from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import make_scorer
import mlflow
import numpy as np

#3.5) Para todos os experimentos o cálculo das métricas (Acurácia, Precision, Recall, Specificity, AUC (classificação);

# Nome do experimento no MLflow
exp_mlflow_ciclo2 = "exp_projeto_ciclo_2"

# Iniciar o tracking do MLflow
mlflow.set_tracking_uri("https://ef00-35-231-71-79.ngrok-free.app/")
mlflow.set_experiment(exp_mlflow_ciclo2)

# Carregar o conjunto de dados
data = load_breast_cancer()
X, y = data.data, data.target

# Inicializar o classificador
classifier = RandomForestClassifier()

# Definir as métricas que deseja calcular
metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'roc_auc': roc_auc_score
}

# Loop sobre as métricas
for metric_name, metric_func in metrics.items():
    # Converter a função de métrica em um scorer
    scorer = make_scorer(metric_func)
    # Calcular as métricas utilizando validação cruzada
    scores = cross_val_score(classifier, X, y, cv=10, scoring=scorer)
    # Calcular a média das métricas
    mean_score = np.mean(scores)
    # Registrar a média das métricas no MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("classifier", classifier.__class__.__name__)
        mlflow.log_metric(f'mean_{metric_name}', mean_score)

        # Salvar o modelo treinado
        mlflow.sklearn.log_model(classifier, "random_forest_model")

    print(f'Mean {metric_name}: {mean_score:.4f}')
