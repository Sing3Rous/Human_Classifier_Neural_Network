import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

def create_data(dir, categories, imageSize, className, isTest):
    data = []
    for category in categories:
        if (isTest):
            path = os.path.join(dir + "\\" + className + "\\" + "validate", category)
        else:
            path = os.path.join(dir + "\\" + className, category)

        classNum = categories.index(category)

        for image in tqdm(os.listdir(path)):
            try:
                imageArray = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
                normalizedImageArray = cv2.resize(imageArray, (imageSize, imageSize))
                data.append([normalizedImageArray, classNum])

            except Exception:
                pass

    return data

def make_pickle(dir, categories, imageSize, className, isTest):
    data = create_data(dir, categories, imageSize, className, isTest)
    random.shuffle(data)
    xData, yData = process_data(data, imageSize)
    if (isTest):
        pickleOut = open(dir + "\\" + className + "\\pickles\\" + "x" + className + "Validate" + ".pickle", "wb")
    else:
        pickleOut = open(dir + "\\" + className + "\\pickles\\" + "x" + className + ".pickle", "wb")
    pickle.dump(xData, pickleOut)
    pickleOut.close()
    if (isTest):
        pickleOut = open(dir + "\\" + className + "\\pickles\\" + "y" + className + "Validate" + ".pickle", "wb")
    else:
        pickleOut = open(dir + "\\" + className + "\\pickles\\" + "y" + className + ".pickle", "wb")
    pickle.dump(yData, pickleOut)
    pickleOut.close()

def process_data(data, imageSize):
    X = []
    y = []
    for features, label in data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, imageSize, imageSize, 3)
    X = X / 255.0
    return X, y

genderCategories = ["male", "female"]
hairColorCategories = ["blonde", "dark"]
ageCategories = ["adult", "child"]

dir = "C:\\Users\\singe\\Documents\\Human Classifier"
imageSize = 200
make_pickle(dir, genderCategories, imageSize, "gender", False)
make_pickle(dir, hairColorCategories, imageSize, "hair_color", False)
make_pickle(dir, ageCategories, imageSize, "age", False)
make_pickle(dir, genderCategories, imageSize, "gender", True)
make_pickle(dir, hairColorCategories, imageSize, "hair_color", True)
make_pickle(dir, ageCategories, imageSize, "age", True)