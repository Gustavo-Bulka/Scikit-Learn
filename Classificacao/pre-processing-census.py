import pandas as pd

#Carregamento da database
base = pd.read_csv('census.csv')

#Divisão da database
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

#Transformação das variáveis categóricas 

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_previsores = LabelEncoder()


previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1]) 
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3]) 
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5]) 
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6]) 
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7]) 
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8]) 
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9]) 
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13]) 

#Transformação final pelo método de Dummy
onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

#Transformação da classe para 0 e 1 
labelencoder_classe = LabelEncoder()
classe  = labelencoder_classe.fit_transform(classe)

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
previsoresTreinamento, previsoresTeste, classeTreinamento , classeTeste = train_test_split(previsores,classe,test_size=0.15,random_state = 0)

