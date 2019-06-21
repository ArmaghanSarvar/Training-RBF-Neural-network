## Training-RBF-neural-network
using ES algorithm to train RBF network and implement regression and classification on dataset


in this project python DEAP library has been used in order to get access to Evolution Strategy algorithm.
actually the fitness of evolutionary algorithm has been calculated using the RBF network.

the activation function of the network(G), guessed output(guessedY), weights' matrix(W) and finally the loss which is
our fitness have been calculated and modified during the algorithm. ( relations available in the definition.pdf file )

### for regression we have:
X(our data) = n * d 
Y = n * 1
W = m * 1 (m = numOfClusters)
G = n * m

the output of regdata2000 training mode:
