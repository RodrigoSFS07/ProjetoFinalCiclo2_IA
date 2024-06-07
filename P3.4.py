from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

#Para contornar erros de carregamento de modelos de machine learning, como "undefined" (3.4), optei por implementar os modelos diretamente no código-fonte.
#essa solução foi escolhida como último recurso para garantir o desenvolvimento contínuo do sistema.
#embora essa abordagem possa ter algumas limitações, como menor flexibilidade e escalabilidade, ela foi documentada de forma clara para facilitar a
#compreensão e manutenção do código por outros desenvolvedores.

class OLA(BaseEstimator, ClassifierMixin):

    # Inicializa o classificador OLA
    def __init__(self, base_estimator=None):
        # Atribui o estimador base
        self.base_estimator = base_estimator

    # Método para treinar o classificador
    def fit(self, X, y):
        # Verifica e ajusta os dados de entrada
        X, y = check_X_y(X, y)
        # Obtém as classes únicas dos rótulos
        self.classes_ = np.unique(y)
        # Lista para armazenar os estimadores treinados para cada classe
        self.estimators_ = []
        # Itera sobre as classes e treina um estimador para cada uma delas
        for target_class in self.classes_:
            # Treina o estimador base apenas nos dados pertencentes à classe atual
            estimator = self.base_estimator.fit(X[y == target_class], y[y == target_class])
            # Armazena o estimador treinado na lista
            self.estimators_.append(estimator)
        # Retorna o próprio objeto
        return self

    # Método para realizar a predição
    def predict(self, X):
        # Verifica se o classificador foi treinado
        check_is_fitted(self)
        # Verifica e ajusta os dados de entrada
        X = check_array(X)

        # Matriz para armazenar as votações de cada estimador
        votes = np.zeros((X.shape[0], len(self.estimators_)))
        # Itera sobre os estimadores e faz a predição para cada um deles
        for i, estimator in enumerate(self.estimators_):
            votes[:, i] = estimator.predict(X)

        # Calcula a predição final utilizando votação majoritária
        y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=votes)
        # Retorna os rótulos preditos
        return y_pred

class LCA(BaseEstimator, ClassifierMixin):

    # Inicializa o classificador LCA
    def __init__(self, base_estimator=None, k_neighbors=3):
        # Atribui o estimador base e o número de neighbors
        self.base_estimator = base_estimator
        self.k_neighbors = k_neighbors

    # Método para treinar o classificador
    def fit(self, X, y):
        # Verifica e ajusta os dados de entrada
        X, y = check_X_y(X, y)
        # Obtém as classes únicas dos rótulos
        self.classes_ = np.unique(y)
        # Lista para armazenar os estimadores treinados para cada classe
        self.estimators_ = []
        # Inicializa o modelo de k-neighbors mais próximos
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        # Itera sobre as classes e treina um estimador para cada uma delas
        for target_class in self.classes_:
            # Treina o estimador base apenas nos dados pertencentes à classe atual
            estimator = self.base_estimator.fit(X[y == target_class], y[y == target_class])
            # Armazena o estimador treinado na lista
            self.estimators_.append(estimator)
            # Ajusta o modelo de k-neighbors mais próximos com os dados da classe atual
            self.nn_.fit(X[y == target_class])
        # Retorna o próprio objeto
        return self

    # Método para realizar a predição
    def predict(self, X):
        # Verifica se o classificador foi treinado
        check_is_fitted(self)
        # Verifica e ajusta os dados de entrada
        X = check_array(X)

        # Matriz para armazenar as votações de cada estimador
        votes = np.zeros((X.shape[0], len(self.estimators_)))
        # Itera sobre os estimadores e calcula as votações
        for i, estimator in enumerate(self.estimators_):
            # Calcula as distâncias para os k-vizinhos mais próximos
            dists, _ = self.nn_.kneighbors(X)
            # Soma as distâncias, excluindo o vizinho mais próximo (ele mesmo)
            votes[:, i] = np.sum(dists[:, 1:], axis=1)

        # Realiza a predição baseada na menor soma de distâncias
        y_pred = np.argmin(votes, axis=1)
        # Retorna os rótulos preditos
        return y_pred

class KNORAU(BaseEstimator, ClassifierMixin):

    # Inicializa o classificador KNORAU
    def __init__(self, base_estimator=None, k_neighbors=3):
        # Atribui o estimador base e o número de vizinhos
        self.base_estimator = base_estimator
        self.k_neighbors = k_neighbors

    # Método para treinar o classificador
    def fit(self, X, y):
        # Verifica e ajusta os dados de entrada
        X, y = check_X_y(X, y)
        # Obtém as classes únicas dos rótulos
        self.classes_ = np.unique(y)
        # Lista para armazenar os estimadores treinados para cada classe
        self.estimators_ = []
        # Itera sobre as classes e treina um estimador para cada uma delas
        for target_class in self.classes_:
            # Treina o estimador base apenas nos dados pertencentes à classe atual
            estimator = self.base_estimator.fit(X[y == target_class], y[y == target_class])
            # Armazena o estimador treinado na lista
            self.estimators_.append(estimator)
        # Retorna o próprio objeto
        return self

    # Método para realizar a predição
    def predict(self, X):
        # Verifica se o classificador foi treinado
        check_is_fitted(self)
        # Verifica e ajusta os dados de entrada
        X = check_array(X)

        # Matriz para armazenar as votações de cada estimador
        votes = np.zeros((X.shape[0], len(self.estimators_)))
        # Itera sobre os estimadores e calcula as votações
        for i, estimator in enumerate(self.estimators_):
            # Calcula as distâncias para os k-vizinhos mais próximos
            dists, _ = estimator.kneighbors(X, n_neighbors=self.k_neighbors)
            # Soma as distâncias
            votes[:, i] = np.sum(dists, axis=1)

        # Realiza a predição baseada no índice da menor soma de distâncias
        y_pred = np.argmin(votes, axis=1)
        # Retorna os rótulos preditos
        return y_pred

class KNORAE(BaseEstimator, ClassifierMixin):

    # Inicializa o classificador KNORAE
    def __init__(self, base_estimator=None, k_neighbors=3):
        # Atribui o estimador base e o número de vizinhos
        self.base_estimator = base_estimator
        self.k_neighbors = k_neighbors

    # Método para treinar o classificador
    def fit(self, X, y):
        # Verifica e ajusta os dados de entrada
        X, y = check_X_y(X, y)
        # Obtém as classes únicas dos rótulos
        self.classes_ = np.unique(y)
        # Lista para armazenar os estimadores treinados para cada classe
        self.estimators_ = []
        # Itera sobre as classes e treina um estimador para cada uma delas
        for target_class in self.classes_:
            # Treina o estimador base apenas nos dados pertencentes à classe atual
            estimator = self.base_estimator.fit(X[y == target_class], y[y == target_class])
            # Armazena o estimador treinado na lista
            self.estimators_.append(estimator)
        # Retorna o próprio objeto
        return self

    # Método para realizar a predição
    def predict(self, X):
        # Verifica se o classificador foi treinado
        check_is_fitted(self)
        # Verifica e ajusta os dados de entrada
        X = check_array(X)

        # Matriz para armazenar as votações de cada estimador
        votes = np.zeros((X.shape[0], len(self.estimators_)))
        # Itera sobre os estimadores e calcula as votações
        for i, estimator in enumerate(self.estimators_):
            # Calcula as distâncias para os k-vizinhos mais próximos
            dists, _ = estimator.kneighbors(X, n_neighbors=self.k_neighbors)
            # Soma as distâncias
            votes[:, i] = np.sum(dists, axis=1)

        # Realiza a predição baseada no índice da maior soma de distâncias
        y_pred = np.argmax(votes, axis=1)
        # Retorna os rótulos preditos
        return y_pred

class MCB(BaseEstimator, ClassifierMixin):

    # Inicializa o classificador MCB
    def __init__(self, base_estimator=None):
        # Atribui o estimador base
        self.base_estimator = base_estimator

    # Método para treinar o classificador
    def fit(self, X, y):
        # Verifica e ajusta os dados de entrada
        X, y = check_X_y(X, y)
        # Obtém as classes únicas dos rótulos
        self.classes_ = np.unique(y)
        # Lista para armazenar os estimadores treinados para cada classe
        self.estimators_ = []
        # Itera sobre as classes e treina um estimador para cada uma delas
        for target_class in self.classes_:
            # Treina o estimador base apenas nos dados pertencentes à classe atual
            estimator = self.base_estimator.fit(X[y == target_class], y[y == target_class])
            # Armazena o estimador treinado na lista
            self.estimators_.append(estimator)
        # Retorna o próprio objeto
        return self

    # Método para realizar a predição
    def predict(self, X):
        # Verifica se o classificador foi treinado
        check_is_fitted(self)
        # Verifica e ajusta os dados de entrada
        X = check_array(X)

        # Matriz para armazenar as votações de cada estimador
        votes = np.zeros((X.shape[0], len(self.estimators_)))
        # Itera sobre os estimadores e faz a predição para cada um deles
        for i, estimator in enumerate(self.estimators_):
            votes[:, i] = estimator.predict(X)

        # Calcula a predição final utilizando votação majoritária
        y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=votes)
        # Retorna os rótulos preditos
        return y_pred

# Nome do experimento no MLflow
exp_mlflow_ciclo2 = "exp_projeto_ciclo_2"

# Iniciar o tracking do MLflow
mlflow.set_tracking_uri("https://ef00-35-231-71-79.ngrok-free.app/")
mlflow.set_experiment(exp_mlflow_ciclo2)

# Carregar o conjunto de dados
data = load_breast_cancer()
X, y = data.data, data.target

# Função para treinar e avaliar o classificador
def train_and_evaluate_classifier(classifier, X, y):
    # Treinar o classificador
    classifier.fit(X, y)

    # Fazer previsões no conjunto de dados
    y_pred = classifier.predict(X)

    # Calcular a acurácia
    accuracy = accuracy_score(y, y_pred)
    return accuracy

# Função que vai rodar experimentos para um modelo específico
def run_experiment(model, model_params, algorithm_name, selection_methods=None):
    for i, param_set in enumerate(model_params):
        with mlflow.start_run(run_name=f"{algorithm_name}_var{i+1}"):
            mlflow.log_params(param_set)
            model_instance = model(**param_set)
            scores = cross_val_score(model_instance, X, y, cv=10)
            mean_accuracy = scores.mean()
            mlflow.log_metric("accuracy", mean_accuracy)
            print(f"{algorithm_name} - Variação {i+1}: Acurácia média = {mean_accuracy:.4f}")

            # Log do modelo treinado
            mlflow.sklearn.log_model(model_instance, f"{algorithm_name}_model_var{i+1}")

        if selection_methods is not None:
            # Executar experimentos de seleção dinâmica
            for method in selection_methods:
                with mlflow.start_run(run_name=f"{algorithm_name}_{method.__class__.__name__}_var{i+1}"):
                    mlflow.log_params({"selection_method": method.__class__.__name__})
                    model_instance = model(**param_set)
                    scores = cross_val_score(model_instance, X, y, cv=10)
                    mean_accuracy = scores.mean()
                    mlflow.log_metric("accuracy", mean_accuracy)
                    print(f"{algorithm_name} com {method.__class__.__name__}: Acurácia média = {mean_accuracy:.4f}")

                    # Log do modelo treinado
                    mlflow.sklearn.log_model(model_instance, f"{algorithm_name}_{method.__class__.__name__}_model_var{i+1}")

# Definir os parâmetros padrão para Bagging e RandomForest
bagging_default_params = {"n_estimators": 10}
random_forest_default_params = {"n_estimators": 100}

# Definir os modelos
models = [
    (BaggingClassifier, bagging_default_params, "Bagging"),
    (RandomForestClassifier, random_forest_default_params, "RandomForest")
]

# Definir os métodos de seleção dinâmica
selection_methods = [
    OLA(),
    LCA(),
    KNORAU(),
    KNORAE(),
    MCB()
]

# Rodar os experimentos para cada modelo e método de seleção dinâmica
for model, model_params, algorithm_name in models:
    print(f"\n{algorithm_name}:")
    run_experiment(model, [model_params], algorithm_name, selection_methods)