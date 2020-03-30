import pandas as pd

#Importando a base de dados 
base = pd.read_csv('credit-data.csv')
#Substituindo os valores de idade negativa pela media 
base.loc[base.age < 0, 'age'] = 40.92
               
#Separando os previsores e as classes 
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

#Substituindo os valores faltantes péla media 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

#Deixando os valores na mesma escala // Sem o escalonamento a precisa caiu 20%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#Dividindo a base de dados em treinamento e teste 
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#Chamando o algoritmo KNN 
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#Verificando a precisao do algoritmo  
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

import collections
collections.Counter(classe_teste)