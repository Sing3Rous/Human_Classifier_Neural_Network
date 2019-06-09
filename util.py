import os
import pickle
import json
import datetime
from tensorflow.keras.models import model_from_json

#returns list of train and list of validate pickles
def load_data(dir):
    path = os.path.join(dir, "categories")
    data = []
    validateData = []
    for className in os.listdir(path):
        path = os.path.join(dir, "categories")
        pickles = []
        validatePickles = []
        path = os.path.join(path, className, "pickles")

        with open(os.path.join(path, "x" + className + "Validate" + ".pickle"), "rb") as validatePickleInX:
            validatePickles.append(pickle.load(validatePickleInX))
        with open(os.path.join(path, "y" + className + "Validate" + ".pickle"), "rb") as validatePickleInY:
            validatePickles.append(pickle.load(validatePickleInY))
        with open(os.path.join(path, "x" + className + ".pickle"), "rb") as pickleInX:
            pickles.append(pickle.load(pickleInX))
        with open(os.path.join(path, "y" + className + ".pickle"), "rb") as pickleInY:
            pickles.append(pickle.load(pickleInY))

        data.append(pickles)
        validateData.append(validatePickles)

    return data, validateData

#saves model by two files: .json and .h5(weights)
#model name: %class name%_conv[%x%]_filters[%y%]_batches[%z%], where
#x - number of convolutional layers, y - number of filters, z - number of batches
def save_model(model, dir, CNNArchitecture, CNNparameters):
    modelJson = model.to_json()
    modelName = CNNparameters["className"] + "_conv[" + str(CNNArchitecture["numOfConvLayers"]) \
                + "]_" + "filters[" + str(CNNArchitecture["filters"]) + "]_" + "batches[" + str(CNNparameters["batchSize"]) \
                + "]"
    path = os.path.join(dir, "categories", CNNparameters["className"], "models", modelName)
    if (os.path.exists(path)):
        now = datetime.datetime.now()
        path += "_("  + str(now.hour) + "_" + str(now.minute) + "_" + str(now.microsecond) + ")"
    os.mkdir(path)
    with open(os.path.join(path, modelName + ".json"), 'w') as jsonFile:
        jsonFile.write(modelJson)
    model.save_weights(os.path.join(path, modelName + ".h5"))

#returns model by reading two files: .json and .h5(weights)
def load_model(modelName, dir, className=""):
    if (className == ""):
        path = os.path.join(dir, "validation", "models", modelName)
    else:
        path = os.path.join(dir, "categories", className, "models", modelName)
    with open(os.path.join(path, modelName + ".json"), 'r') as jsonFile:
        modelJson = jsonFile.read()
    model = model_from_json(modelJson)
    model.load_weights(os.path.join(path, modelName + ".h5"))
    return model

#returns list of all cnn architectures
def load_cnn_architectures(dir):
    architectures = []
    path = os.path.join(dir, "categories")
    for categoryPath in os.listdir(path):
        with open(os.path.join(path, categoryPath, "architecture.json"), "r") as file:
            architecture = json.load(file)
            architectures.append(architecture)

    return architectures

#returns list of all cnn parameters
def load_cnn_parameters_list(dir):
    parametersList = []
    path = os.path.join(dir, "categories")
    for categoryPath in os.listdir(path):
        with open(os.path.join(path, categoryPath, "parameters.json"), "r") as file:
            parameters = json.load(file)
            parametersList.append(parameters)

    return parametersList

