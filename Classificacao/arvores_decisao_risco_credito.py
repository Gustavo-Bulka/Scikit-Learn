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


# importando a arvore de decisao do sklearn
from sklearn.tree import DecisionTreeClassifier
#Montando a arvore de decisao com criterio de entropia como visto na teoria 
#porem temos outros criterios que podem ser mudados olhando no CTRL+ G
classificador = DecisionTreeClassifier(criterion = 'entropy')
classificador.fit(previsores,classe)
# Printando a importancia de cada coluna para a arvore de decisao 
print(classificador.feature_importances_)

# Vizualizando a nossa arvore de decisão 
from sklearn.tree import export
export.export_graphviz(classificador,
                       out_file ='arvore.dot', 
                       feature_names = ['Historia','Divida', 'Garantias', 'Renda'],
                       class_names = ['Alto', 'Moderado', 'Baixo'],
                       filled = True,
                       leaves_parallel = True)

