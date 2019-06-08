import time
import os
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from util import load_data
from util import save_model
from util import load_cnn_architectures
from util import load_cnn_parameters_list

def main():
    dir = "C:\\Users\\singe\\Documents\\Human Classifier"

    data, validateData = load_data(dir)
    CNNArchitectures = load_cnn_architectures(dir)
    CNNParametersList = load_cnn_parameters_list(dir)

    # process logs
    boards = []
    for i in range(len(data)):
        logName = (CNNParametersList[i]["className"] + "-{}-conv-{}-filters-{}-batches-{}").format(CNNArchitectures[i]["numOfConvLayers"],
                                                                     CNNArchitectures[i]["filters"],
                                                                     CNNParametersList[i]["batchSize"],
                                                                     int(time.time()))
        board = TensorBoard(
            log_dir=(os.path.join(dir, "categories", CNNParametersList[i]["className"], "logs") + "{}").format(logName))
        boards.append(board)

    # build & train models
    for i in range(len(data)):
        model = build_model(CNNArchitectures[i], data[i])
        train_model(model, CNNParametersList[i], data[i], validateData[i], boards[i])
        save_model(model, dir, CNNArchitectures[i], CNNParametersList[i])



def add_convolutional_layers(numOfLayers, model, filters, kernelSize, poolSize):
    for i in range(numOfLayers):
        model.add(Conv2D(filters, (kernelSize, kernelSize)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

def add_dense_layers(numOfLayers, model, units):
    for i in range(numOfLayers):
        model.add(Dense(units))
        model.add(Activation('relu'))

def add_input_layer(X, model, filters, kernelSize, poolSize):
    model.add(Conv2D(filters, (kernelSize, kernelSize), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

def add_output_layer(model):
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

#builds model by the following architecture
def build_model(CNNArchitecture, trainData):
    model = Sequential()
    add_input_layer(trainData[0], model,
                    CNNArchitecture["filters"],
                    CNNArchitecture["kernelSize"],
                    CNNArchitecture["poolSize"])
    add_convolutional_layers(CNNArchitecture["numOfConvLayers"], model,
                             CNNArchitecture["filters"],
                             CNNArchitecture["kernelSize"],
                             CNNArchitecture["poolSize"])
    model.add(Flatten())
    add_dense_layers(CNNArchitecture["numOfDenseLayers"], model, CNNArchitecture["units"])
    add_output_layer(model)
    return model

#trains model by the following parameters
def train_model(model, CNNparameters, trainData, validateData, board):
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(trainData[0], trainData[1],
              batch_size=CNNparameters["batchSize"],
              epochs=CNNparameters["numOfEpochs"],
              validation_split=0.3,
              callbacks=[board],
              validation_data=validateData)

if __name__ == "__main__":
    main()