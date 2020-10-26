import torch
import numpy as np
import matplotlib.pyplot as plt


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

    def forward(self, trainingsData):
        inputToHidden = self.input(trainingsData)
        hiddenOneToHiddenTwo = self.hiddenOne(self.activation(inputToHidden))
        hiddenTwoToHiddenThree = self.hiddenTwo(
            self.activation(hiddenOneToHiddenTwo))
        hiddenThreeToHiddenfour = self.hiddenThree(
            self.activation(hiddenTwoToHiddenThree))
        hiddenFourToOutput = self.hiddenFour(
            self.activation(hiddenThreeToHiddenfour))
        yPrediction = self.output(hiddenFourToOutput)
        return yPrediction


def createPlotsamples(spatialRefinement):
    xSamples = np.linspace(0,
                           2. * np.pi,
                           spatialRefinement,
                           endpoint=True,
                           dtype=np.float32)
    plotSamples = np.zeros(shape=(spatialRefinement, 2), dtype=np.float32)
    plotSamples[:, 0] = 0.
    plotSamples[:, 1] = xSamples
    plotData = torch.from_numpy(plotSamples)
    return plotData

def exactInitialData(x):
    return 1. - np.cos(x)

dimIn, dimHiddenOne, dimHiddenTwo, dimHiddenThree, dimHiddenFour, dimOut, nSamples = 2, 5, 5, 5, 5, 1, 100

model = FFN(dimIn, dimHiddenOne, dimHiddenTwo, dimHiddenThree, dimHiddenFour,
            dimOut)

model = torch.load(
    "/media/fm/2881fd19-010f-4d7b-a148-a8973130f331/fabian/pytorch_coding/PINN/model.pt"
)

model.eval()
plotData = createPlotsamples(10000)
y = np.zeros(10000)
yExact = np.zeros(10000)

for i in range(10000):
    y[i] = model(plotData[i, :])
    yExact[i] = exactInitialData(plotData[i, 1].double())

plt.plot(plotData[:, 1], y, label='NN initial condition')
plt.plot(plotData[:, 1], yExact, label='exact initial condition')

plt.show()
