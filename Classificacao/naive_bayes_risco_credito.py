import pandas as pd

#Carregando a base de dados
base = pd.read_csv("risco-credito.csv")

#Separando a base em previsores e classe
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

#Transformando os dados de testo para dados numericos
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in range(4):
    previsores[:,i] = labelencoder.fit_transform(previsores[:,i])


# importando o naive bayes do sklearn
from sklearn.naive_bayes import GaussianNB
#Montando a tabela de probabilidade 
classificador = GaussianNB()
classificador.fit(previsores,classe)

#Vamos ver o resultado do cliente novo abaixo pra comparar com a teoria 
#Historia BOA//// Divida ALTA//// garantia NENHUMA//// renda>35

resultado = classificador.predict([[0,0,1,2]])
print("O risco do Cliente eh: ")
print(resultado)

