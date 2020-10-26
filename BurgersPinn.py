import torch
import numpy as np
import random
import os


def exactInitialCondition(x):
    return 1. - np.cos(x)


def createTrainingsamples(numberSamples, spatialRefinement,
                          temporalRefinement):
    xSamples = np.linspace(0,
                           2. * np.pi,
                           spatialRefinement,
                           endpoint=False,
                           dtype=np.float32)
    tSamples = np.linspace(0,
                           0.5,
                           temporalRefinement,
                           endpoint=False,
                           dtype=np.float32)
    initialT = 0.
    xLeftBoundary = 0.
    xRightBoundary = 2. * np.pi

    initialConditionSamples = np.zeros(shape=(numberSamples, 2),
                                       dtype=np.float32)
    initialConditionSamples[:, 0] = initialT
    initialConditionSamples[:, 1] = xSamples[np.random.randint(
        0, spatialRefinement, numberSamples)]
    trueInitialcondition = exactInitialCondition(initialConditionSamples[:, 1])

    boundaryConditionSamples = np.zeros(shape=(numberSamples, 2),
                                        dtype=np.float32)
    boundaryConditionSamples[:, 0] = tSamples[np.random.randint(
        0, temporalRefinement, numberSamples)]
    boundaryConditionSamples[:, 1] = random.choices(
        [xLeftBoundary, xRightBoundary], k=numberSamples)

    trainingData = np.append(initialConditionSamples,
                             boundaryConditionSamples,
                             axis=0)
    trainingData = torch.from_numpy(trainingData)
    trainingData.requires_grad = True
    referenceData = np.append(trueInitialcondition,
                              np.zeros(numberSamples, dtype=np.float32),
                              axis=0)
    referenceData = torch.from_numpy(referenceData)
    return trainingData, referenceData


def createCollocationPoints(nSamples, spatialRefinement, temporalRefinement):
    xSamples = np.linspace(0 + 1. / spatialRefinement,
                           2. * np.pi - 1. / spatialRefinement,
                           spatialRefinement,
                           endpoint=False,
                           dtype=np.float32)
    tSamples = np.linspace(0 + 1. / temporalRefinement,
                           0.5 - 1. / temporalRefinement,
                           temporalRefinement,
                           endpoint=False,
                           dtype=np.float32)
    collocationPoints = np.zeros(shape=(nSamples, 2), dtype=np.float32)
    collocationPoints[:, 0] = random.choices(tSamples, k=nSamples)
    collocationPoints[:, 1] = random.choices(xSamples, k=nSamples)
    collocationtrainingData = torch.from_numpy(collocationPoints)
    collocationtrainingData.requires_grad = True
    return collocationtrainingData


# simple feedforward net with 4 hidden layers and tanh activation function
class FFN(torch.nn.Module):
    def __init__(self, dimIn, dimHiddenOne, dimHiddenTwo, dimHiddenThree,
                 dimHiddenFour, dimOut):
        super(FFN, self).__init__()
        self.input = torch.nn.Linear(dimIn, dimHiddenOne)
        self.hiddenOne = torch.nn.Linear(dimHiddenOne, dimHiddenTwo)
        self.hiddenTwo = torch.nn.Linear(dimHiddenTwo, dimHiddenThree)
        self.hiddenThree = torch.nn.Linear(dimHiddenThree, dimHiddenFour)
        self.hiddenFour = torch.nn.Linear(dimHiddenFour, dimOut)
        self.output = torch.nn.Linear(dimOut, 1)
        self.activation = torch.nn.Tanh()

    def forward(self, trainingData):
        inputToHidden = self.input(trainingData)
        hiddenOneToHiddenTwo = self.hiddenOne(self.activation(inputToHidden))
        hiddenTwoToHiddenThree = self.hiddenTwo(
            self.activation(hiddenOneToHiddenTwo))
        hiddenThreeToHiddenfour = self.hiddenThree(
            self.activation(hiddenTwoToHiddenThree))
        hiddenFourToOutput = self.hiddenFour(
            self.activation(hiddenThreeToHiddenfour))
        yPrediction = self.output(hiddenFourToOutput)
        return yPrediction


def residual(trainingData):
    u = model(trainingData)
    gradient = torch.autograd.grad(u,
                                   trainingData,
                                   create_graph=True,
                                   allow_unused=True)
    dtU = gradient[0][0]
    uSquare = 0.5 * u * u
    gradientUSquare = torch.autograd.grad(uSquare,
                                          trainingData,
                                          create_graph=True,
                                          allow_unused=True)
    dxUSquare = gradientUSquare[0][1]
    return dtU + dxUSquare


dimIn, dimHiddenOne, dimHiddenTwo, dimHiddenThree, dimHiddenFour, dimOut, nSamples = 2, 5, 5, 5, 5, 1, 100
trainingData, yTrue = createTrainingsamples(nSamples, 1000, 1000)
collocationData = createCollocationPoints(1000, 1000, 1000)

lossFunction = torch.nn.MSELoss(reduction='sum')

if (os.path.isfile(
        "/media/fm/2881fd19-010f-4d7b-a148-a8973130f331/fabian/pytorch_coding/PINN/model.pt"
)):
    model = torch.load(
        "/media/fm/2881fd19-010f-4d7b-a148-a8973130f331/fabian/pytorch_coding/PINN/model.pt"
    )
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
else:
    model = FFN(dimIn, dimHiddenOne, dimHiddenTwo, dimHiddenThree,
                dimHiddenFour, dimOut)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    modelErrors = np.zeros(500)
    for iterations in range(500):
        epochLoss = 0.
        shuffeledList = list(range(2 * nSamples))
        random.shuffle(shuffeledList)
        for sample in shuffeledList:
            randomSample = np.random.randint(0, 2 * nSamples)
            yPrediction = model(trainingData[randomSample, :])
            loss = lossFunction(yPrediction, yTrue[randomSample])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

        print("Epoch: {0}, Loss :{1}".format(iterations, epochLoss))
        modelErrors[iterations] = epochLoss
    np.savetxt("modelErrors.txt", modelErrors)

    torch.save(
        model,
        "/media/fm/2881fd19-010f-4d7b-a148-a8973130f331/fabian/pytorch_coding/PINN/model.pt"
    )

residualErrors = np.zeros(1000)
nSamplesResidual = 1000

for iterations in range(100):
    epochLoss = 0.
    for sample in range(nSamplesResidual):
        randomSample = np.random.randint(0, nSamplesResidual)
        yPrediction = residual(collocationData[randomSample, :])
        optimizer.zero_grad()
        yTrue = torch.zeros(1)[0]
        lossResidual = lossFunction(yPrediction, yTrue)
        lossResidual.backward(retain_graph=True)
        optimizer.step()
        epochLoss += lossResidual.item()

    print("Epoch: {0}, Loss :{1}".format(iterations, epochLoss))
    residualErrors[iterations] = epochLoss
np.savetxt("ResidualErrors.txt", residualErrors)

torch.save(
    model,
    "/media/fm/2881fd19-010f-4d7b-a148-a8973130f331/fabian/pytorch_coding/PINN/fullBurgers.pt"
)
