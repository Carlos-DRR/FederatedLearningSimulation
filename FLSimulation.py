import numpy as np
import random

listaPontosExemplo1 = [[1, 2], [2, 2], [3, 3], [3, 4]]
listaPontosExemplo2 = [[4, 2], [3, 2], [3, 1], [3, 0], [1, 2]]
listaPontosExemplo3 = [[5, 2], [5, 1], [2, 0], [1, 2], [4, 1], [3, 2]]
eta = 0.01

class Model():

  def __init__(self, taxaDeAprendizagem, epocas):
    self.a = 1 # coeficiente angular
    self.b = 0 # termo independente
    self.eta = taxaDeAprendizagem
    self.epocas = epocas

  def getEpocas(self):
    return self.epocas

  def getEta(self):
    return self.eta

  def getCoeficientes(self):
    return [self.a, self.b]

  def train(self, listaPontos):
    #equacao do modelo = ax + b
    custo = 0
    for x in range(0, self.epocas):
      for ponto in listaPontos:
        x = ponto[0]
        y = ponto[1]
        y_pred = self.a * x + self.b
        erro =  y - y_pred
        grad_a = (-2 * erro) * x
        grad_b = (-2 * erro)
        self.a = self.a - self.eta * grad_a
        self.b = self.b - self.eta * grad_b
        custo = erro
    
    self.custo = erro

  def getCusto(self):
    return self.custo
  def setA(self, a):
    self.a = a
  
  def setB(self, b):
    self.b = b 

  def test(self, x):
    return (self.a * x) + self.b


class Cliente():
  def __init__(self, model, listaPontos):
    self.listaPontos = listaPontos
    self.qtdDados = len(listaPontos)
    self.model = model

  def getQtdDados(self):
    return self.qtdDados

  def getListaPontos(self):
    return self.listaPontos

  def setModel(self, model):
    self.model = model
  
  def getModel(self):
    return self.model

  def trainModel(self):
    self.model.train(self.listaPontos)

  def getModelCustoFinal(self):
    return self.model.getCusto()

  


class Servidor():

  def __init__(self, listaClientes, modeloGlobal, iteracoes):
    self.listaClientes = listaClientes
    self.modeloGlobal = modeloGlobal
    self.iteracoes = iteracoes

  def getGlobalModel(self):
      return self.modeloGlobal

  def train(self):
    for i in range(0, self.iteracoes):
      subConjuntoClientes = random.sample(self.listaClientes, 2)
      listaUpdates = []
      totalDados = 0
      novoModeloCoef = np.array([0, 0])
      for cliente in subConjuntoClientes:
        cliente.setModel(self.modeloGlobal)
        cliente.trainModel()

        print("Custo Local {}".format(cliente.getModelCustoFinal()))
        totalDados += cliente.getQtdDados()
        listaUpdates.append([cliente.getModel(), cliente.getQtdDados()])

      for update in listaUpdates:
        modeloLocal = update[0]
        qtdDadosLocal = update[1]
        coef = np.array(modeloLocal.getCoeficientes())
        novoModeloCoef = novoModeloCoef + (coef * (qtdDadosLocal/totalDados))
      
      novoModelo = Model(self.modeloGlobal.getEta(), self.modeloGlobal.getEpocas())
      novoModelo.setA(novoModeloCoef[0])
      novoModelo.setB(novoModeloCoef[1])
      self.modeloGlobal = novoModelo
      print(self.modeloGlobal.getCoeficientes())

#eta, epocas
model_c1 = Model(0.01,1000)
model_c2 = Model(0.01,1000)
model_c3 = Model(0.01,1000)

#modelo, lista de pontos
cliente1 = Cliente(model_c1, listaPontosExemplo1)
cliente2 = Cliente(model_c2, listaPontosExemplo2)
cliente3 = Cliente(model_c3, listaPontosExemplo3)

listaClientes = [cliente1, cliente2, cliente3]

modeloGlobalInicial = Model(0.001, 1000)

servidor = Servidor(listaClientes, modeloGlobalInicial, 7)

servidor.train()
modeloFederado = servidor.getGlobalModel()
print("Modelo federado para a equação f(x) = a * x + b ")
print("f(x) = " + str(modeloFederado.getCoeficientes()[0]) + " x" + " + " + str(modeloFederado.getCoeficientes()[1]))
