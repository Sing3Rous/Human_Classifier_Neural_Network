import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

def create_data(dir, categories, imageSize):
    data = []
    for category in categories:
        path = os.path.join(dir, category)
        classNum = categories.index(category)

        for image in tqdm(os.listdir(path)):
            try:
                imageArray = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
                normalizedImageArray = cv2.resize(imageArray, (imageSize, imageSize))
                data.append([normalizedImageArray, classNum])

            except Exception:
                pass

    return data

def make_pickle(dir, categories, imageSize, className):
    data = create_data(dir, categories, imageSize)
    random.shuffle(data)
    xData, yData = process_data(data, imageSize)
    pickleOut = open(dir + "\\" + className + "\\pickles\\" + "x" + className + ".pickle", "wb")
    pickle.dump(xData, pickleOut)
    pickleOut.close()
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
skinColorCategories = ["bright", "dark"]

dir = "C:\\Users\\singe\\Documents\\Human Classifier"
imageSize = 200
make_pickle(dir, genderCategories, imageSize, "gender")
make_pickle(dir, hairColorCategories, imageSize, "hair_color")
make_pickle(dir, skinColorCategories, imageSize, "age")