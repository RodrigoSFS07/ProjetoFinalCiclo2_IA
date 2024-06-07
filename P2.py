import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset
dataset = pd.read_csv('C:/Users/Rodrigo/Downloads/archive/data.csv')

# Exibir informações sobre o conjunto de dados
print('\nInformações sobre o Dataset:\n')
print(dataset.info())

# Verificar se há valores nulos no conjunto de dados
print('\nValores Nulos no Dataset:\n')
print(dataset.isnull().sum())

# Exibir as primeiras linhas do conjunto de dados
print('\nAmostra do Dataset Original:\n')
print(dataset.head())

# Converter os labels para números
dataset['diagnosis'] = dataset['diagnosis'].astype('category')
dataset['diagnosis'] = dataset['diagnosis'].cat.codes

# Remover coluna 'Unnamed: 32' (caso esteja presente)
if 'Unnamed: 32' in dataset.columns:
    dataset = dataset.drop(columns=['Unnamed: 32'])

# Normalizar as features na escala 0..1
scaler = MinMaxScaler()
columns_to_normalize = dataset.columns[1:]  # Todas as colunas exceto 'id' e 'diagnosis'
dataset[columns_to_normalize] = scaler.fit_transform(dataset[columns_to_normalize])

# Exibir o dataset tratado
print('\nDataset Tratado:\n')
print(dataset.head())