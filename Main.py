import array
import random
import matplotlib.pyplot as plot

import numpy as np

import math
import time
import csv
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

test = True   # test or train? change this
scale = 100
MIN_VALUE = 0
MAX_VALUE = 5 # of data
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3
numOfClusters = 10 # m
numOfClasses = 1 # for regression = 1
dimension = 3 # d
numOfData = 2000 # L
# this is the number of training data in train mode
IND_SIZE = numOfClusters * (dimension + 1)
# radial of cluster 1, dim1 of cluster1, dim 2 of cluster 1, radial of cluster 1, dim1 of cluster2,...

def initialize():
    global data, labels, gMatrix, wMatrix, guessedY, classificationOutput, colors, numOfAllData
    data = np.random.rand(numOfData, dimension)
    labels = np.zeros((numOfData, numOfClasses))
    gMatrix = np.random.rand(numOfData, numOfClusters)
    wMatrix = np.random.rand(numOfClusters, numOfClasses)  # m * c
    guessedY = np.random.rand(numOfData, numOfClasses)
    classificationOutput = np.zeros((numOfData,1))
    colors = ['red', 'pink', 'yellow', 'magenta', 'blue', 'black', 'green']  # to the num of classes
    if numOfClasses == 1 and test is False:
        numOfAllData = input("Enter the numOfAllData : ")


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(IND_SIZE))
    # gammas and their clusters in each chromosome
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(IND_SIZE))
    return ind


def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator


def fillData():
    global labels, data
    datax = []
    with open('regdata2000.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        rowNum = 0
        for row in readCSV:
            datax.append(list(map(float, row)))
            rowNum += 1
    datax = np.array(datax)
    tempY = datax[:, datax.shape[1] - 1]
    tempY = tempY/scale
    tempX = datax[:, :datax.shape[1] - 1]
    tempX = tempX/scale
    # interval = 1      # when classification
    interval = int(numOfAllData) / (numOfData + 1)   # when regression and train  #TODO numOfTrainData < numOfData
    classes = []
    for i in range(0, numOfData):
        for j in range(0, dimension):    # fill data (x)
            data[i][j] = tempX[int(i * interval)][j]
        if numOfClasses == 1:
            labels[i][0] = tempY[int(i * interval)]  # TODO
        else:    # fill labels(y) = numOfData * numOfClasses
            for j in range(0, numOfClasses):
                if not classes.__contains__(tempY[int(i * interval)]):
                    classes.append(tempY[int(i * interval)])
                if j == classes.index(tempY[int(i * interval)]):
                    labels[i][j] = 1
                else:
                    labels[i][j] = 0
    #print(labels)
    # print(data)

def fillG(individual):
    for i in range(0, numOfData):
        for j in range(0, numOfClusters):
            gamma = 0
            for d in range(0, dimension):
                gamma += (data[i][d] - individual[d + 1 + j * (dimension + 1)])**2
            gMatrix[i][j] = math.exp(-gamma * individual[j * (dimension + 1)] / scale)


def fillW():
    global wMatrix
    mul = np.matmul(gMatrix.transpose(), gMatrix)
    mul = np.linalg.inv(mul)
    mul = np.matmul(mul, gMatrix.transpose())
    wMatrix = np.matmul(mul, labels)


def calculateCError():  # when classification
    error = 0
    for i in range(0, numOfData):
        if labels[i][int(classificationOutput[i])] == 0:
            error += 1
    return 100 * error / numOfData


def calculateError():   # when regression
    error = 0
    for i in range(0, numOfData):
        for j in range(0, numOfClasses):
            error += (guessedY[i][j] - labels[i][j]) ** 2
    return error / 2


def calculateFitness(individual):
    try:
        fillG(individual)
        global guessedY
        guessedY = np.matmul(gMatrix, wMatrix)
        error = calculateError()
        fillW()
    except:
        return math.inf,
    if math.isnan(error):
        return math.inf,

    return error,

######################################################################################

toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                 IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selBest)
toolbox.register("evaluate", calculateFitness)
toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))


def saveLearned(path, matrix, row, col):
    f = open(path, 'w+')
    if f.mode == 'w+':
        for i in range(0, row):
            for j in range(0, col):
                f.write("%f " % matrix[i][j])
            f.write("\n")
    f.close()


def loadLearned(path, matrix):
    sample = 0
    f = open(path, 'r')
    if f.mode == 'r':
        lines = f.readlines()
        for line in lines:
            line = line[:-2]
            splits = line.split(' ')
            for i in range(0, len(splits)):
                matrix[sample][i] = float(splits[i])
            sample += 1
    f.close()
    return matrix


def main():
    global guessedY, chosenChromosome
    random.seed()
    MU, LAMBDA = 10, 100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=0.6, mutpb=0.3, ngen=10, stats=stats, halloffame=hof)

    chosenChromosome = pop[0]   # we have the best chromosome in final pop
    fillG(pop[0])
    guessedY = np.matmul(gMatrix, wMatrix)

    saveLearned("current_generation.txt", pop, MU, IND_SIZE)
    saveLearned("current_weights.txt", wMatrix, numOfClusters, numOfClasses)

    return pop, logbook, hof


chosenChromosome = []
def testF():
    global wMatrix, guessedY, chosenChromosome
    MU, LAMBDA = 10, 100
    pop = toolbox.population(n=MU)
    pop = loadLearned("current_generation.txt", pop)
    wMatrix = loadLearned("current_weights.txt", wMatrix)

    chosenChromosome = pop[0]     # converged chromosome
    fillG(pop[0])
    guessedY = np.matmul(gMatrix, wMatrix)


def plotting():
    if numOfClasses == 1:
        for i in range(0, numOfData):
            plot.scatter(i, labels[i][0], color='green')
            plot.scatter(i, guessedY[i][0], color='red')
        print('Error is:', calculateError())
        plot.show()
    elif numOfClasses > 1 and dimension == 2:
        for i in range(0, numOfData):
            maxIndex = 0
            for j in range(0, numOfClasses):
                if guessedY[i][j] > guessedY[i][maxIndex]:
                    maxIndex = j
            classificationOutput[i] = maxIndex
            if labels[i][int(classificationOutput[i])] == 0:
                plot.scatter(data[i][0], data[i][1], color='orange')  # wrong classes
            else:
                plot.scatter(data[i][0], data[i][1], color=colors[int(classificationOutput[i])])  # other classes
        for i in range(0, numOfClusters):
            plot.scatter(chosenChromosome[i * (dimension + 1) + 1]/scale,
                         chosenChromosome[i * (dimension + 1) + 2]/scale,
                         color='green')

            circle = plot.Circle((chosenChromosome[i * (dimension + 1) + 1] / scale,
                         chosenChromosome[i * (dimension + 1) + 2]/scale), 1/math.sqrt(scale * math.fabs(chosenChromosome[i * (dimension + 1)])),edgecolor='black', facecolor='none')
            aaa = plot.gca()
            aaa.add_patch(circle)
            # plot.axis('scaled')
        print('Error is:', calculateCError(), '%')
        print('َAccuracy is:', 100 - calculateCError(), '%')
        plot.show()


if __name__ == "__main__":
    initialize()
    fillData()
    start = time.time()

    if not test:
        main()
    else:
        testF()
    print('time taken =', str(time.time() - start))
    plotting()
