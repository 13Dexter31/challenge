import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)

nrows = 10000
nsteps = 5000
timeseries = np.zeros((nrows, nsteps, 1))
t = np.linspace(0, 30, num=nsteps)

for i in range(nrows):
    a = 2 * (np.random.rand(11) - 0.5)
    b = 2 * (np.random.rand(11) - 0.5)
    timeseries[i, :, 0] = a[0] / 2

    for k in range(1, 11):
        timeseries[i, :, 0] += a[k] * np.cos(t * k)
        timeseries[i, :, 0] += b[k] * np.sin(t * k)

    timeseries[i, :, 0] = 2 * timeseries[i, :, 0] / np.max(np.abs(timeseries[i, :, 0]))

Y = np.zeros(nrows)
yTrue = np.random.choice(nrows, int(nrows * 0.2), replace=False)
Y[yTrue] = 1
stoerung = np.array([0.1, 0.2, 0.4, 0.4, 0.3, 0.4, 0.4, 0.2, 0, 0, 0, 0, 0, 0, 0, 0,
                     -0.1, -0.2, -0.4, -0.2, -0.2, -0.4, -0.2, -0.1] * 2)
for i in yTrue:
    xpos = np.random.randint(0, nsteps - len(stoerung))
    timeseries[i, xpos:xpos + len(stoerung), 0] += stoerung

TrainSet = np.random.choice(timeseries.shape[0], int(timeseries.shape[0] * 0.80), replace=False)
XTrain = timeseries[TrainSet, :]
YTrain = Y[TrainSet]

TestSet = np.delete(np.arange(len(Y)), TrainSet)
XTest = timeseries[TestSet, :]
YTest = Y[TestSet]

sampleWeight = np.ones_like(YTrain)
sampleWeight[YTrain == 0] = 0.3

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
import tensorflow as tf

tf.random.set_seed(42)

model = Sequential()
model.add(Conv1D(8, kernel_size=10, activation='relu', use_bias=False,
                 input_shape=(timeseries.shape[1], 1)))

model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(6, kernel_size=8, activation='relu', use_bias=False))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(1, kernel_size=4, activation='relu', use_bias=False))
model.add(MaxPooling1D(pool_size=4))

model.add(Flatten())
# model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(XTrain, YTrain, epochs=30, verbose=True, sample_weight=sampleWeight)
model.summary()
print(model.evaluate(XTest, YTest, verbose=False))

plt.figure()
plt.plot(history.history['loss'], color='k', label='loss', marker='*')
plt.plot(history.history['accuracy'], color='r', label='accuracy', marker='+')
plt.legend()

from keras import backend as K


def zwischenVerarbeitungPlotten(sigSingle, outLayer):
    signal = sigSingle[np.newaxis, ...]
    outputSingleLayer = K.function([model.layers[0].input], [model.layers[outLayer].output])
    myout = outputSingleLayer([signal])[0]

    plt.figure()
    plt.subplot(myout.shape[2] + 1, 1, 1)
    plt.xlim([0, nsteps])
    plt.title('Original Signal')
    plt.plot(sigSingle.squeeze(), linewidth=2, c='r')
    for i in range(myout.shape[2]):
        plt.subplot(myout.shape[2] + 1, 1, i + 2)

    plt.plot(myout[0, :, i].squeeze(), linewidth=2, c='k')
    plt.xlim([0, len(myout[0, :, i].squeeze())])
    mystring = 'Effect of Learned Filter ' + str(i)
    plt.title(mystring)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8)
    plt.show()


firstY0 = np.flatnonzero(np.logical_not(YTest))[0]
firstY1 = np.flatnonzero(YTest)[0]
yP = model.predict(XTest)
print(yP[firstY0], yP[firstY1])
zwischenVerarbeitungPlotten(XTest[firstY0, :], 0)
zwischenVerarbeitungPlotten(XTest[firstY1, :], 0)
