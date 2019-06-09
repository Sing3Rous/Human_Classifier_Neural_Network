import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

def main():
    dir = "C:\\Users\\singe\\Documents\\Human Classifier"
    imageSize = 200

    make_pickles(dir, imageSize)

#takes each image in folder and makes an data array of every image
#returns an array of arrays of data of images
def create_data(dir, categories, imageSize, isValidate=False):
    data = []
    for category in categories:
        path = dir
        if (isValidate):
            path = os.path.join(dir, "validate", category)
        else:
            path = os.path.join(dir, "train", category)

        classNum = categories.index(category)
        for image in tqdm(os.listdir(path)):
            imageArray = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
            normalizedImageArray = cv2.resize(imageArray, (imageSize, imageSize))
            data.append([normalizedImageArray, classNum])

    return data

#makes pickle of images in folder
def make_pickles(dir, imageSize):
    path = os.path.join(dir, "categories")

    for category in os.listdir(path):
        path = os.path.join(dir, "categories")
        categories = os.listdir(os.path.join(path, category, "train"))

        for isValidate in [False, True]:
            path = os.path.join(dir, "categories")
            data = create_data(os.path.join(path, category), categories, imageSize, isValidate)
            random.shuffle(data)
            xData, yData = process_data(data, imageSize)
            path = os.path.join(path, category, "pickles")
            if (not os.path.exists(path)):
                os.mkdir(path)
            if (isValidate):
                pickleOutX = open(os.path.join(path, "x" + category + "Validate" + ".pickle"), "wb")
                pickleOutY = open(os.path.join(path, "y" + category + "Validate" + ".pickle"), "wb")
            else:
                pickleOutX = open(os.path.join(path, "x" + category + ".pickle"), "wb")
                pickleOutY = open(os.path.join(path, "y" + category + ".pickle"), "wb")
            pickle.dump(xData, pickleOutX)
            pickle.dump(yData, pickleOutY)
            pickleOutX.close()
            pickleOutY.close()

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