from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

# Nome do experimento no MLflow
exp_mlflow_ciclo2 = "exp_projeto_ciclo_2"

# Iniciar o tracking do MLflow
mlflow.set_tracking_uri("https://ef00-35-231-71-79.ngrok-free.app/")
mlflow.set_experiment(exp_mlflow_ciclo2)

# Carregar o conjunto de dados
data = load_breast_cancer()
X, y = data.data, data.target

# Definir o método de seleção de features
selector = SelectKBest(score_func=f_classif, k=1)  # definindo o número de features
method_name = 'SelectKBest_f_classif'  # Nome do método de seleção de features

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar o método de seleção de features aos dados de treinamento e teste
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Função para treinar e avaliar o classificador
def train_and_evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    # Treinar o classificador com as features selecionadas
    classifier.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = classifier.predict(X_test)

    # Calcular a acurácia
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Lista de modelos de classificação
models = [
    RandomForestClassifier(),
    # Outros modelos podem ser adicionados aqui
]

# Executar os experimentos para cada modelo
for model in models:
    with mlflow.start_run():
        # Treinar e avaliar o classificador com seleção de features
        accuracy = train_and_evaluate_classifier(model, X_train_selected, X_test_selected, y_train, y_test)

        # Registrar a acurácia no MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("feature_selection_method", method_name)

        # Log do modelo treinado
        mlflow.sklearn.log_model(model, "model")

        print(f'Acurácia ({method_name}): {accuracy:.4f}')