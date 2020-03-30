import pandas as pd

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Não é necessario dividir em treinamento e teste nesse caso pois 
# a validacao cruzada ja vai dividir a base de acordo com o K que 
# passarmos a ela no caso cv=10 
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()


resultados = cross_val_score(classificador, previsores, classe, cv = 10)
# a precisao do algoritmo é dada pela media dos resuldados dados pelo cross_val_score
resultados.mean()
resultados.std()

