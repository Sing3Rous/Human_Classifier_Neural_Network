from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import model_from_json
import pickle
import time
import os

def main():
    # cnn architectures:
    ####################################################################################################################
    genderCNNArchitecture = {
        'numOfConvLayers': 2,
        'filters': 32,
        'kernelSize': 3,
        'numOfDenseLayers': 0,
        'units': 0,
        'poolSize': 2
    }

    ageCNNArchitecture = {
        'numOfConvLayers': 2,
        'filters': 32,
        'kernelSize': 3,
        'numOfDenseLayers': 0,
        'units': 0,
        'poolSize': 2
    }

    hairColorCNNArchitecture = {
        'numOfConvLayers': 2,
        'filters': 32,
        'kernelSize': 3,
        'numOfDenseLayers': 0,
        'units': 0,
        'poolSize': 2
    }
    ####################################################################################################################

    # cnn parameters
    ####################################################################################################################
    genderCNNparameters = {
        'batchSize': 16,
        'numOfEpochs': 5
    }

    ageCNNparameters = {
        'batchSize': 16,
        'numOfEpochs': 5
    }

    hairColorCNNparameters = {
        'batchSize': 16,
        'numOfEpochs': 5
    }
    ####################################################################################################################
    genderClassName = "gender"
    ageClassName = "age"
    hairColorClassName = "hair_color"

    dir = "C:\\Users\\singe\\Documents\\Human Classifier"

    genderTrain = load_data(dir, genderClassName, False)
    ageTrain = load_data(dir, ageClassName, False)
    hairColorTrain = load_data(dir, hairColorClassName, False)
    genderTrainValidate = load_data(dir, genderClassName, True)
    ageTrainValidate = load_data(dir, ageClassName, True)
    hairColorTrainValidate = load_data(dir, hairColorClassName, True)

    # process logs
    ####################################################################################################################
    genderLogName = "gender-{}-conv-{}-filters-{}-batches-{}".format(genderCNNArchitecture["numOfConvLayers"],
                                                                     genderCNNArchitecture["filters"],
                                                                     genderCNNparameters["batchSize"],
                                                                     int(time.time()))
    genderBoard = TensorBoard(
        log_dir='C:\\Users\\singe\\Documents\\Human Classifier\\gender\\logs\\{}'.format(genderLogName))

    ageLogName = "age-{}-conv-{}-filters-{}-batches-{}".format(ageCNNArchitecture["numOfConvLayers"],
                                                               ageCNNArchitecture["filters"],
                                                               ageCNNparameters["batchSize"],
                                                               int(time.time()))
    ageBoard = TensorBoard(log_dir='C:\\Users\\singe\\Documents\\Human Classifier\\age\\logs\\{}'.format(ageLogName))

    hairColorLogName = "hair_color-{}-conv-{}-filters-{}-batches-{}".format(hairColorCNNArchitecture["numOfConvLayers"],
                                                                            hairColorCNNArchitecture["filters"],
                                                                            hairColorCNNparameters["batchSize"],
                                                                            int(time.time()))
    hairColorBoard = TensorBoard(
        log_dir='C:\\Users\\singe\\Documents\\Human Classifier\\hair_color\\logs\\{}'.format(hairColorLogName))
    ####################################################################################################################

    # build & train models
    ####################################################################################################################
    genderModel = build_model(genderCNNArchitecture, genderTrain)
    train_model(genderModel, genderCNNparameters, genderTrain, genderTrainValidate, genderBoard)
    save_model(genderModel, dir, genderClassName, genderCNNArchitecture, genderCNNparameters)

    ageModel = build_model(ageCNNArchitecture, ageTrain)
    train_model(ageModel, ageCNNparameters, ageTrain, ageTrainValidate, ageBoard)
    save_model(ageModel, dir, ageClassName, ageCNNArchitecture, ageCNNparameters)

    hairColorModel = build_model(hairColorCNNArchitecture, hairColorTrain)
    train_model(hairColorModel, hairColorCNNparameters, hairColorTrain, hairColorTrainValidate, hairColorBoard)
    save_model(hairColorModel, dir, hairColorClassName, hairColorCNNArchitecture, hairColorCNNparameters)
    ####################################################################################################################

def research():
    # research architecture & parameters of cnn
    ####################################################################################################################
    researchNumOfConvLayers = [1, 2, 3]
    researchFilters = [16, 32]
    researchBatchSize = [8, 16, 32]

    researchCNNArchitecture = {
        'numOfConvLayers': 2,
        'filters': 16,
        'kernelSize': 3,
        'numOfDenseLayers': 0,
        'units': 0,
        'poolSize': 2
    }

    researchCNNparameters = {
        'batchSize': 16,
        'numOfEpochs': 8
    }
    ####################################################################################################################
    genderClassName = "gender"
    ageClassName = "age"
    hairColorClassName = "hair_color"

    dir = "C:\\Users\\singe\\Documents\\Human Classifier"

    genderTrain = load_data(dir, genderClassName, False)
    ageTrain = load_data(dir, ageClassName, False)
    hairColorTrain = load_data(dir, hairColorClassName, False)
    genderTrainValidate = load_data(dir, genderClassName, True)
    ageTrainValidate = load_data(dir, ageClassName, True)
    hairColorTrainValidate = load_data(dir, hairColorClassName, True)

    # research code
    ####################################################################################################################
    for numOfConvLayers in researchNumOfConvLayers:
        for filters in researchFilters:
            for batchSize in researchBatchSize:
                researchCNNArchitecture.pop("numOfConvLayers")
                researchCNNArchitecture.pop("filters")
                researchCNNparameters.pop("batchSize")
                researchCNNArchitecture.setdefault("numOfConvLayers", numOfConvLayers)
                researchCNNArchitecture.setdefault("filters", filters)
                researchCNNparameters.setdefault("batchSize", batchSize)

                genderLogName = "gender-{}-conv-{}-filters-{}-batches-{}".format(
                    researchCNNArchitecture["numOfConvLayers"],
                    researchCNNArchitecture["filters"],
                    researchCNNparameters["batchSize"],
                    int(time.time()))
                genderBoard = TensorBoard(
                    log_dir='C:\\Users\\singe\\Documents\\Human Classifier\\gender\\logs\\{}'.format(genderLogName))

                ageLogName = "age-{}-conv-{}-filters-{}-batches-{}".format(researchCNNArchitecture["numOfConvLayers"],
                                                                           researchCNNArchitecture["filters"],
                                                                           researchCNNparameters["batchSize"],
                                                                           int(time.time()))
                ageBoard = TensorBoard(
                    log_dir='C:\\Users\\singe\\Documents\\Human Classifier\\age\\logs\\{}'.format(ageLogName))

                hairColorLogName = "hair_color-{}-conv-{}-filters-{}-batches-{}".format(
                    researchCNNArchitecture["numOfConvLayers"],
                    researchCNNArchitecture["filters"],
                    researchCNNparameters["batchSize"],
                    int(time.time()))
                hairColorBoard = TensorBoard(
                    log_dir='C:\\Users\\singe\\Documents\\Human Classifier\\hair_color\\logs\\{}'.format(
                        hairColorLogName))

                genderModel = build_model(researchCNNArchitecture, genderTrain)
                train_model(genderModel, researchCNNparameters, genderTrain, genderTrainValidate, genderBoard)
                save_model(genderModel, dir, genderClassName, researchCNNArchitecture, researchCNNparameters)

                ageModel = build_model(researchCNNArchitecture, ageTrain)
                train_model(ageModel, researchCNNparameters, ageTrain, ageTrainValidate, ageBoard)
                save_model(ageModel, dir, ageClassName, researchCNNArchitecture, researchCNNparameters)

                hairColorModel = build_model(researchCNNArchitecture, hairColorTrain)
                train_model(hairColorModel, researchCNNparameters, hairColorTrain, hairColorTrainValidate,
                            hairColorBoard)
                save_model(hairColorModel, dir, hairColorClassName, researchCNNArchitecture, researchCNNparameters)
    ####################################################################################################################

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

#returns a pickle of images
def load_data(datadir, className, isTest):
    pickles = []
    path = os.path.join(datadir, "categories", className, "pickles")
    if (isTest):
        pickleIn = open(os.path.join(path, "x" + className + "Validate" + ".pickle"), "rb")
    else:
        pickleIn = open(os.path.join(path, "x" + className + ".pickle"), "rb")
    pickles.append(pickle.load(pickleIn))
    if (isTest):
        pickleIn = open(os.path.join(path, "y" + className + "Validate" + ".pickle"), "rb")
    else:
        pickleIn = open(os.path.join(path, "y" + className + ".pickle"), "rb")
    pickles.append(pickle.load(pickleIn))
    return pickles

#saves model by two files: .json and .h5(weights)
#model name: %class name%_conv[%x%]_filters[%y%]_batches[%z%], where
#x - number of convolutional layers, y - number of filters, z - number of batches
def save_model(model, datadir, className, CNNArchitecture, CNNparameters):
    modelJson = model.to_json()
    path = os.path.join(datadir, "categories", className, "models")
    with open(os.path.join(path, className + "_conv[" + str(CNNArchitecture["numOfConvLayers"])
              + "]_" + "filters[" + str(CNNArchitecture["filters"]) + "]_" + "batches[" + str(CNNparameters["batchSize"])
              + "]" + ".json", 'w')) as jsonFile:
        jsonFile.write(modelJson)
    model.save_weights(os.path.join(path, className + "_conv["
                       + str(CNNArchitecture["numOfConvLayers"]) + "]_" + "filters[" + str(CNNArchitecture["filters"])
                       + "]_" + "batches[" + str(CNNparameters["batchSize"]) + "]" + ".h5"))

#returns model by reading two files: .json and .h5(weights)
def load_model(modelName, datadir, className):
    path = os.path.join(datadir, "categories", className, "models")
    jsonFile = open(os.path.join(path, modelName + ".json"), 'r')
    modelJson = jsonFile.read()
    jsonFile.close()
    model = model_from_json(modelJson)
    model.load_weights(os.path.join(path, modelName + ".h5"))
    return model

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
    #research()