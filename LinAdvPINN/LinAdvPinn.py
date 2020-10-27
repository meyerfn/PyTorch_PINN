import torch
import numpy as np
import random
import os


def exactInitialCondition(x):
    return 1. - np.cos(x)


def exactBoundaryCondition(data):
    t = data[:, 0]
    x = data[:, 1]
    return 1. - np.cos(x - t)


def createTrainingsamples(nSamples):
    xSamples = np.linspace(0,
                           2. * np.pi,
                           nSamples,
                           endpoint=True,
                           dtype=np.float32)
    np.random.shuffle(xSamples)
    tSamples = np.linspace(0, 1.0, nSamples, endpoint=True, dtype=np.float32)
    np.random.shuffle(tSamples)

    initialT = 0.
    xLeftBoundary = 0.
    xRightBoundary = 2. * np.pi

    initialConditionSamples = np.zeros(shape=(nSamples, 2), dtype=np.float32)
    initialConditionSamples[:, 0] = initialT
    initialConditionSamples[:, 1] = xSamples

    boundaryConditionSamples = np.zeros(shape=(nSamples, 2), dtype=np.float32)
    boundaryConditionSamples[:, 0] = tSamples
    boundaryConditionSamples[:, 1] = random.choices(
        [xLeftBoundary, xRightBoundary], k=nSamples)

    trueInitialcondition = exactInitialCondition(initialConditionSamples[:, 1])
    trueBoundaryData = exactBoundaryCondition(boundaryConditionSamples)

    trainingData = np.append(initialConditionSamples,
                             boundaryConditionSamples,
                             axis=0)
    trainingData = torch.from_numpy(trainingData)
    trainingData.requires_grad = True

    referenceData = np.append(trueInitialcondition, trueBoundaryData, axis=0)
    referenceData = torch.from_numpy(referenceData)
    return trainingData, referenceData


def createCollocationPoints(nSamples):
    xSamples = np.linspace(0,
                           2. * np.pi,
                           nSamples,
                           endpoint=False,
                           dtype=np.float32)
    np.random.shuffle(xSamples)
    tSamples = np.linspace(0., 1.0, nSamples, endpoint=False, dtype=np.float32)
    np.random.shuffle(tSamples)

    collocationPoints = np.zeros(shape=(nSamples, 2), dtype=np.float32)

    collocationPoints[:, 0] = tSamples
    collocationPoints[:, 1] = xSamples

    collocationtrainingData = torch.from_numpy(collocationPoints)
    collocationtrainingData.requires_grad = True
    return collocationtrainingData


# simple feedforward net with 5 hidden layers and tanh activation function
class FFN(torch.nn.Module):
    def __init__(self, dimIn, dimHidden, dimOut, numberHiddenLayers):
        super(FFN, self).__init__()
        self.input = torch.nn.Linear(dimIn, dimHidden)
        self.hidden = torch.nn.Linear(dimHidden, dimHidden)
        self.hiddenToOutput = torch.nn.Linear(dimHidden, dimOut)
        self.output = torch.nn.Linear(dimOut, 1)
        self.activation = torch.nn.Tanh()

    def forward(self, trainingData):
        outputFromInputLayer = self.activation(self.input(trainingData))
        for i in range(numberHiddenLayers - 1):
            outputFromHiddenLayer = self.activation(
                self.hidden(outputFromInputLayer))
        outputFromHiddenLayer = self.activation(
            self.hiddenToOutput(outputFromHiddenLayer))
        yPrediction = self.output(outputFromHiddenLayer)
        return yPrediction

    def residual(self, trainingData):
        u = model(trainingData)
        gradient = torch.autograd.grad(outputs=u,
                                       inputs=trainingData,
                                       grad_outputs=torch.ones_like(u),
                                       create_graph=True,
                                       allow_unused=True)[0]
        dtU = gradient[:, 0]
        dxU = gradient[:, 1]
        return dtU + dxU


dimIn, dimHidden, dimOut, numberHiddenLayers = 2, 10, 1, 5
batchSize, batchSizeCollocation, nSamples, nSamplesResidual = 10, 100, 100, 1000

trainingData, trueOutput = createTrainingsamples(nSamples)
collocationData = createCollocationPoints(nSamplesResidual)

lossFunction = torch.nn.MSELoss()
model = FFN(dimIn, dimHidden, dimOut, numberHiddenLayers)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)
epochErrors = np.zeros(1000)

for iterations in range(1000):
    epochLoss = 0.
    shuffeledList = list(range(10))
    random.shuffle(shuffeledList)
    for i in shuffeledList:
        uPrediction = model(trainingData[i * batchSize:(i + 1) * batchSize, :])
        lossU = lossFunction(uPrediction[:, 0],
                             trueOutput[i * batchSize:(i + 1) * batchSize])
        residualPrediction = model.residual(
            collocationData[i * batchSizeCollocation:(i + 1) *
                            batchSizeCollocation, :])
        trueResidual = torch.zeros(100)
        lossResidual = lossFunction(trueResidual, residualPrediction)
        combinedLoss = lossU + lossResidual

        optimizer.zero_grad()
        combinedLoss.backward()
        optimizer.step()

        epochLoss += combinedLoss.item()

    print("Epoch: {0}, Loss :{1}".format(iterations, epochLoss))
    epochErrors[iterations] = epochLoss
    if (np.mod(iterations, 100) == 0):
        torch.save(
            model,
            "/media/fm/2881fd19-010f-4d7b-a148-a8973130f331/fabian/pytorch_coding/PINN/LinAdvPINN/plot/fullModel{0}.pt"
            .format(iterations))
np.savetxt("modelErrors.txt", epochErrors)