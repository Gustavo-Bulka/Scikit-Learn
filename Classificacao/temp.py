#---------------------------------------------------------------------
#Pré Processamento do banco de dados 

import pandas as pd

# Importando a database a partir do arquivo CSV 
base = pd.read_csv('credit-data.csv')

#Substituindo as idades negativas pela média das idades positivas 
base.loc[base['age']<0] = base['age'][base.age>0].mean()

#Separando os previsores da classe
previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values


#Usando sckitlearn para preencher os valores faltantes 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(previsores[:,1:4])
previsores[:,1:4] = imputer.transform(previsores[:,1:4])


#Usando o sckitlearn para escalonar os valores dos previsores 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#---------------------------------------------------------------------
#Divisão da database em Treinamento e Teste 

# A biblioteca Sklearn pode fazer essa divisão automaticamente sem 
#termos a necessidade de escolher quais dados usaremos para cada um 
from sklearn.model_selection import train_test_split

# Criando as variaveis para separação 
# O teste size nos mostra que 25% da base sera teste e o restante sera treinamento 
#para bases grandes usar 10% para pequenas 30%
previsoresTreinamento, previsoresTeste, classeTreinamento , classeTeste = train_test_split(previsores,classe,test_size=0.25,random_state = 0)

# Importando naive bayes

from sklearn.naive_bayes import GaussianNB
classificador  = GaussianNB()
classificador.fit(previsoresTreinamento,classeTreinamento)
resultado = classificador.predict(previsoresTeste)


