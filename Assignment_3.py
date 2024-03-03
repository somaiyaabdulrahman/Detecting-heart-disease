from typing import Counter
from unicodedata import name
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import random

# The dataset is uploaded
f = open("Assignment 3 medical_dataset.data")
dataset_X = []
dataset_y = []
line = " "
while line != "":
    line = f.readline()
    line = line[:-1]
    if line != "":
        line = line.split(",")
        floatList = []
        for i in range(len(line)):
            if i < len(line)-1:
                floatList.append(float(line[i]))
            else:
                value = float(line[i])
                if value == 0:
                    dataset_y.append(0)
                else:
                    dataset_y.append(1)
        dataset_X.append(floatList)
f.close()

# The dataset is splited into training and test.
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size = 0.25, random_state = 0)

# The dataset is scaled
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# The model is created
model = KNeighborsClassifier(n_neighbors = 3)

# Function that calculates the fitness of a solution
def calculateFitness(solution):
    fitness = 0

    # The features are selected according to solution
    X_train_Fea_selc = []
    X_test_Fea_selc = []
    for example in X_train:
        X_train_Fea_selc.append([a*b for a,b in zip(example,solution)])
    for example in X_test:
        X_test_Fea_selc.append([a*b for a,b in zip(example,solution)])

    model.fit(X_train_Fea_selc, y_train)

    # We predict the test cases
    y_pred = model.predict(X_test_Fea_selc)

    # We calculate the Accuracy
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0] # True positives
    FP = cm[0][1] # False positives
    TN = cm[1][1] # True negatives
    FN = cm[1][0] # False negatives

    fitness = (TP + TN) / (TP + TN + FP + FN)

    return round(fitness *100,2)

MAX_FITNESS_CALCULATIONS = 5000

def randomRestart():    #random currentSoultion when wanting to randomrestart
    randomCurrentSolution = []
    for i in range(14):
        randomNumber = random.randint(0,1)
        randomCurrentSolution.append(randomNumber)

    return randomCurrentSolution

def findNeigbours(currentSolution): #changes one element in the current solution and saves it as a neigbour
    allNeighbours = []
    for i in range(len(currentSolution)):
        neigbour = currentSolution.copy()
        neigbour[i] = 1 - neigbour[i]
        allNeighbours.append(neigbour)

    return allNeighbours

def findNeighboursVNS(currentSolution, numberOfChanges): #changes 1/2/3 element in the current solution and saves it as a neigbour
    allNeighboursVNS = []

    if numberOfChanges == 1:
        for i in range(len(currentSolution)):
            neighbour = currentSolution.copy()
            neighbour[i]= 1 - neighbour[i]
            allNeighboursVNS.append(neighbour)

    if numberOfChanges == 2:
        for i in range(len(currentSolution)):
            neighbour = currentSolution.copy()
            neighbour[i]= 1 - neighbour[i]
            allNeighboursVNS.append(neighbour)

            for j in range(i+1,len(currentSolution)):
                neighbour = currentSolution.copy()
                neighbour[j]= 1 - neighbour[j]
                allNeighboursVNS.append(neighbour)

    if numberOfChanges == 3:
        for i in range(len(currentSolution)):
            neighbour = currentSolution.copy()
            neighbour[i]= 1 - neighbour[i]
            allNeighboursVNS.append(neighbour)

            for j in range(i+1,len(currentSolution)):
                neighbour = currentSolution.copy()
                neighbour[j]= 1 - neighbour[j]
                allNeighboursVNS.append(neighbour)

            for l in range(j+1,len(currentSolution)):
                neighbour = currentSolution.copy()
                neighbour[l]= 1 - neighbour[l]
                allNeighboursVNS.append(neighbour)

    return allNeighboursVNS

def hillClimbing(currentSolution):
    FITNESS_CALCULATIONS_COUNTER = 0
    currentSolutionFitness = calculateFitness(currentSolution)
    bestSolution = []
    bestSolutionFitness = 0
    bestNeighbour = []
    bestNeighbourFitness = 0

    while True:
        allNeighbours = findNeigbours(currentSolution)
        for neighbour in allNeighbours:
            neighbourFitness = calculateFitness(neighbour)
            FITNESS_CALCULATIONS_COUNTER += 1

            if neighbourFitness > bestNeighbourFitness:
                bestNeighbourFitness = neighbourFitness
                bestNeighbour = neighbour

        if bestNeighbourFitness > currentSolutionFitness:
            currentSolution = bestNeighbour
            currentSolutionFitness = bestNeighbourFitness
            if currentSolutionFitness > bestSolutionFitness:
                bestSolution = bestNeighbour
                bestSolutionFitness = bestNeighbourFitness
                print("Best solution fitness for hillclimbing: ( ",FITNESS_CALCULATIONS_COUNTER,"/", MAX_FITNESS_CALCULATIONS,"):", bestSolutionFitness)

        else:
            currentSolution = randomRestart()

        if FITNESS_CALCULATIONS_COUNTER >= MAX_FITNESS_CALCULATIONS:
            return bestSolution, bestSolutionFitness

def VNS(currentSolution):
    FITNESS_CALCULATIONS_COUNTER = 0
    currentSolutionFitness = calculateFitness(currentSolution)
    bestSolution = []
    bestSolutionFitness = 0
    bestNeighbour = []
    bestNeighbourFitness = 0
    counter = 0

    while True:
        allNeighboursVNS = findNeighboursVNS(currentSolution,counter)
        for neighbour in allNeighboursVNS:
            neighbourFitness = calculateFitness(neighbour)
            FITNESS_CALCULATIONS_COUNTER += 1

            if neighbourFitness > bestNeighbourFitness:
                bestNeighbourFitness = neighbourFitness
                bestNeighbour = neighbour

        if bestNeighbourFitness > currentSolutionFitness:
            counter = 1
            currentSolution = bestNeighbour
            currentSolutionFitness = bestNeighbourFitness
            if currentSolutionFitness > bestSolutionFitness:
                bestSolution = bestNeighbour
                bestSolutionFitness = bestNeighbourFitness
                print("Best solution fitness for VNS: ( ",FITNESS_CALCULATIONS_COUNTER,"/", MAX_FITNESS_CALCULATIONS,"):", bestSolutionFitness)


        if bestNeighbourFitness < currentSolutionFitness:
            counter += 1
            if counter == 4:
                currentSolution = randomRestart()

        if FITNESS_CALCULATIONS_COUNTER >= MAX_FITNESS_CALCULATIONS:
            return bestSolution, bestSolutionFitness


def main():
    iterationLine = '-' * 23
    headingLine = '*' * 23
    resultHC = 0
    resultVNS = 0

    currentSolution = []
    for i in range(14):
        randomNumber = random.randint(0,1)
        currentSolution.append(randomNumber)

    for i in range(10):

        print('\n', iterationLine, 'ITERATION', i+1, iterationLine,'\n')
        print('\n', headingLine,'HILL-CLIMBING', headingLine,'\n')

        bestSolution, bestSolutionFitness = hillClimbing(currentSolution)
        print('Hillclimbing', bestSolution, bestSolutionFitness)
        resultHC += bestSolutionFitness

        print('\n', headingLine,'VNS', headingLine,'\n')
        bestSolution, bestSolutionFitness = VNS(currentSolution)
        print('Variable neighbour search', bestSolution, bestSolutionFitness)
        resultVNS += bestSolutionFitness

    print('\n',headingLine, 'END OF ITERATION', headingLine,'\n')


    finalResultHillclimbing = 'Average best solution fitness for hillclimbing', resultHC / 10
    finalResultVNS = 'Average best solution fitness forr VNS', resultVNS / 10

    print('\n',headingLine, 'FINAL RESULT', headingLine, '\n')
    print('\n',finalResultHillclimbing,'\n')
    print('\n',finalResultVNS,'\n')
    print('\n', headingLine, 'END OF RESULT', headingLine,'\n')


main()
