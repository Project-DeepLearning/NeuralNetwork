import os, time, numpy
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as pyplot

cdef class NeuralNetwork:
    cdef object model
    cdef object x_train
    cdef object y_train
    cdef object training
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    def __cinit__(self, x_train: list, y_train: list, epochs: int = 1000):
        self.model = keras.Sequential()
        self.x_train = x_train
        self.y_train = y_train
        self.__convert_input__()
        self.__deep_learning__()
        self.__training__(epochs)

    cdef void __convert_input__(self):
        self.x_train = numpy.array(self.x_train, dtype=numpy.float64)
        self.y_train = numpy.array(self.y_train, dtype=numpy.float64)

    cdef void __deep_learning__(self):
        self.model.add(layers.Dense(units=4096, activation="linear"))
        self.model.add(layers.Dense(units=2048, activation="linear"))
        self.model.add(layers.Dense(units=1, activation="linear"))
        cdef object optimizer = optimizers.Adam(learning_rate=0.01, amsgrad=True)
        self.model.compile(optimizer=optimizer, loss="MSE", metrics=["binary_accuracy"])

    cdef void __training__(self, epochs: int):
        self.training = self.model.fit(self.x_train, self.y_train, epochs=epochs)

    cdef object predict(self, x_predict: object):
        x_predict = numpy.array(x_predict, dtype=numpy.float64)
        return self.model.predict(x_predict)

    cdef void display(self):
        cdef object history_loss = self.training.history
        pyplot.figure(u"Display", figsize = (9, 4))
        pyplot.plot(history_loss["loss"], "r-")
        pyplot.legend(['Loss'], loc='upper left')
        pyplot.title("History")
        pyplot.show()

cpdef public void Main(argv: str):
    x_train = [[1, 1], [2, 2], [4, 1], [0, 0], [1, 3], [2, 1], [4, 5], [0, 3]]
    y_train = [2, 4, 5, 0, 4, 3, 9, 3]
    Network = NeuralNetwork(x_train, y_train)
    x_predict = [[5, 5], [4, 4]]
    y_predict = Network.predict(x_predict)
    print(y_predict)
