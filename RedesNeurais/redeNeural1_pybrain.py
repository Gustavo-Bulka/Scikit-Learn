# Vamos criar a rede neural da figura 

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer,SigmoidLayer,BiasUnit
from pybrain.structure import FullConnection

# Criando a rede 
rede = FeedForwardNetwork()

# Craindo os modulos da nossa rede 

# Camada de entrada com 2 neuronios
camadaEntrada = LinearLayer(2)
# Camada de saida com 3 neuronios 
camadaSaida = SigmoidLayer(3)
# Camada oculta
camadaOculta = SigmoidLayer(1)
# Criando os bias 
biasOculto = BiasUnit()
biasSaida = BiasUnit()

# Colocando os modulos dentro da nossa rede 

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(biasOculto)
rede.addModule(biasSaida)

# Linkando as partes da rede 

entradaOculta = FullConnection(camadaEntrada,camadaOculta)
ocultaSaida = FullConnection(camadaOculta,camadaSaida)
biasOculta = FullConnection(biasOculto,camadaOculta)
biasesaida =  FullConnection(biasSaida,camadaSaida)

#Rodqando a rede
rede.sortModules()

print(rede)

