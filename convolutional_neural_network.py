from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import model_from_json
import pickle
import time

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

def load_data(datadir, className):
    pickles = []
    pickleIn = open(datadir + "\\" + className + "\\" + "pickles\\" + "x" + className + ".pickle", "rb")
    pickles.append(pickle.load(pickleIn))
    pickleIn = open(datadir + "\\" + className + "\\" + "pickles\\" + "y" + className + ".pickle", "rb")
    pickles.append(pickle.load(pickleIn))
    return pickles

def save_model(model, datadir, className):
    modelJson = model.to_json()
    with open(datadir + "\\" + className + "\\models\\" + className + ".json", 'w') as jsonFile:
        jsonFile.write(modelJson)
    model.save_weights(datadir + "\\" + className + "\\models\\" + className + ".h5")

def load_model(datadir, modelName):
    jsonFile = open(datadir + "\\models\\" + modelName + ".json", 'r')
    modelJson = jsonFile.read()
    jsonFile.close()
    model = model_from_json(modelJson)
    model.load_weights(datadir + "\\models\\" + modelName + ".h5")
    return model

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

#cnn architectures:
########################################################################################################################
genderCNNArchitecture = {
    'numOfConvLayers': 3,
    'filters': 64,
    'kernelSize': 3,
    'numOfDenseLayers': 0,
    'units': 0,
    'poolSize': 2
}

ageCNNArchitecture = {
    'numOfConvLayers': 3,
    'filters': 64,
    'kernelSize': 3,
    'numOfDenseLayers': 0,
    'units': 0,
    'poolSize': 2
}

hairColorCNNArchitecture = {
    'numOfConvLayers': 3,
    'filters': 64,
    'kernelSize': 3,
    'numOfDenseLayers': 0,
    'units': 0,
    'poolSize': 2
}
########################################################################################################################

#cnn parameters
########################################################################################################################
genderCNNparameters = {
    'batchSize': 8,
    'numOfEpochs': 3
}

ageCNNparameters = {
    'batchSize': 8,
    'numOfEpochs': 3
}

hairColorCNNparameters = {
    'batchSize': 8,
    'numOfEpochs': 3
}

genderClassName = "gender"
ageClassName = "age"
hairColorClassName = "hair_color"

dir = "C:\\Users\\singe\\Documents\\Human Classifier"
genderTrain = load_data(dir, genderClassName)
ageTrain = load_data(dir, ageClassName)
hairColorTrain = load_data(dir, hairColorClassName)
########################################################################################################################

#process logs
########################################################################################################################
genderLogName = "{}-conv-{}-filters-{}-batches-{}".format(genderCNNArchitecture["numOfConvLayers"],
                                                    genderCNNArchitecture["filters"],
                                                    genderCNNparameters["batchSize"],
                                                    int(time.time()))
genderBoard = TensorBoard(log_dir='C:\\Users\\singe\\Documents\\Human Classifier\\gender\\logs\\{}'.format(genderLogName))

ageLogName = "{}-conv-{}-filters-{}-batches-{}".format(ageCNNArchitecture["numOfConvLayers"],
                                                    ageCNNArchitecture["filters"],
                                                    ageCNNparameters["batchSize"],
                                                    int(time.time()))
ageBoard = TensorBoard(log_dir='C:\\Users\\singe\\Documents\\Human Classifier\\age\\logs\\{}'.format(ageLogName))

hairColorLogName = "{}-conv-{}-filters-{}-batches-{}".format(hairColorCNNArchitecture["numOfConvLayers"],
                                                    hairColorCNNArchitecture["filters"],
                                                    hairColorCNNparameters["batchSize"],
                                                    int(time.time()))
hairColorBoard = TensorBoard(log_dir='C:\\Users\\singe\\Documents\\Human Classifier\\hair_color\\logs\\{}'.format(hairColorLogName))
########################################################################################################################

#build & train models
########################################################################################################################
genderModel = build_model(genderCNNArchitecture, genderTrain)
train_model(genderModel, genderCNNparameters, genderTrain, genderTrain, genderBoard)
save_model(genderModel, dir, genderClassName)

ageModel = build_model(ageCNNArchitecture, ageTrain)
train_model(ageModel, ageCNNparameters, ageTrain, ageTrain, ageBoard)
save_model(ageModel, dir, ageClassName)

hairColorModel = build_model(hairColorCNNArchitecture, hairColorTrain)
train_model(hairColorModel, hairColorCNNparameters, hairColorTrain, hairColorTrain, hairColorBoard)
save_model(hairColorModel, dir, hairColorClassName)
########################################################################################################################