import pandas as pd

base = pd.read_csv('plano-saude.csv')

X = base.iloc[:,0].values
Y = base.iloc[:,1].values

#Vendo a correlação entre ass variaveis 

import numpy as np
correlacao = np.corrcoef(X,Y)

# Para a regressão sckit necessita que trabalhemos 
# com matrizes assim como temos um vetor de uma coluna 
# só temos que transformar em matriz adicionando mais 
# uma coluna sem alterar a primeira
# -1 pra nao mudar as çlinhas e 1 pra adicionar 
X = X.reshape(-1,1)

#fazendo a regressão
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,Y)

#Y = B0+B1x
#Vendo valor de Bo
regressor.intercept_

#Vendo valor de B1
regressor.coef_
