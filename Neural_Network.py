#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

class Neural_Network(object):
  def __init__(self):
    #Define hyperparameters
    self.inputLayerSize = 2
    self.outputLayerSize = 1
    self.hiddenLayerSize = 3
    
    #weights (Paramters)
    self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
    self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

  def forward(self, X):
      #propagate inputs through the network
      self.z2 = np.dot(X, self.W1)
      self.a2 = self.sigmoid(self.z2)
      self.z3 = np.dot(self.a2, self.W2)
      yHat = self.sigmoid(self.z3)
      return yHat
  def sigmoid(self, z):
      #Apply the sigmoid activation function to scalar, vector
      return 1/ (1 + np.exp(-z))
  def sigmoidPrime(z):
      return np.exp(-z)/((1 + np.exp(-z))**2)

def main():
   network = Neural_Network()

   X = [[3,5],[5,1],[10,2]]
   print network.forward(X)
   plt.plot([1,2,3,4])
   plt.ylabel('some numbers')
   plt.show() 

main()
