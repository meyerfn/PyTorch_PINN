import torch
import numpy as np
import matplotlib.pyplot as plt


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


def createPlotsamples(spatialRefinement, tEval):
    xSamples = np.linspace(0,
                           2. * np.pi,
                           spatialRefinement,
                           endpoint=True,
                           dtype=np.float32)
    plotSamples = np.zeros(shape=(spatialRefinement, 2), dtype=np.float32)
    plotSamples[:, 0] = tEval
    plotSamples[:, 1] = xSamples
    plotData = torch.from_numpy(plotSamples)
    return plotData

def exactSolution(data):
    t = data[0]
    x = data[1]
    return 1. - np.cos(x-t)


dimIn, dimHidden, dimOut, numberHiddenLayers = 2, 10, 1, 5
model = FFN(dimIn, dimHidden, dimOut, numberHiddenLayers)

model = torch.load(
    "/media/fm/2881fd19-010f-4d7b-a148-a8973130f331/fabian/pytorch_coding/PINN/LinAdvPINN/plot/fullModel900.pt"
)

model.eval()
tEval = 0.5
y = np.zeros(1000)
yExact = np.zeros(1000)

plotData = createPlotsamples(1000,tEval)
for i in range(1000):
    y[i] = model(plotData[i, :])
    yExact[i] = exactSolution(plotData[i, :].numpy())

plt.plot(plotData[:, 1], y, label= 'NN approximation, t=0')
plt.plot(plotData[:, 1], yExact, label= 'Exact soultion, t=0')

tEval = 1.0
plotData = createPlotsamples(1000,tEval)
for i in range(1000):
    y[i] = model(plotData[i, :])
    yExact[i] = exactSolution(plotData[i, :].numpy())

plt.plot(plotData[:, 1], y, label= 'NN approximation, t=0.5')
plt.plot(plotData[:, 1], yExact, label= 'Exact soultion, t = 0.5')
plt.legend()

plt.show()
