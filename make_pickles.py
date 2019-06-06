import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

def main():
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

#takes each image in folder and makes an data array of every image
#returns an array of arrays of data of images
def create_data(dir, categories, imageSize, className, isTest):
    data = []
    for category in categories:
        if (isTest):
            path = os.path.join(dir, "categories", className, "validate", category)
        else:
            path = os.path.join(dir, "categories", className, category)

        classNum = categories.index(category)

        for image in tqdm(os.listdir(path)):
            try:
                imageArray = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
                normalizedImageArray = cv2.resize(imageArray, (imageSize, imageSize))
                data.append([normalizedImageArray, classNum])

            except Exception:
                pass

    return data

#makes pickle of images in folder
def make_pickle(dir, categories, imageSize, className, isValidate):
    data = create_data(dir, categories, imageSize, className, isValidate)
    random.shuffle(data)
    xData, yData = process_data(data, imageSize)
    path = os.path.join(dir, "categories", className, "pickles")
    if (isValidate):
        pickleOut = open(os.path.join(path, "x" + className + "Validate" + ".pickle"), "wb")
    else:
        pickleOut = open(os.path.join(path, "x" + className + ".pickle"), "wb")
    pickle.dump(xData, pickleOut)
    pickleOut.close()
    if (isValidate):
        pickleOut = open(os.path.join(path, "y" + className + "Validate" + ".pickle"), "wb")
    else:
        pickleOut = open(os.path.join(path, "y" + className + ".pickle"), "wb")
    pickle.dump(yData, pickleOut)
    pickleOut.close()

#normalize data of images
def process_data(data, imageSize):
    X = []
    y = []
    for features, label in data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, imageSize, imageSize, 3)
    X = X / 255.0
    return X, y

if __name__ == "__main__":
    main()