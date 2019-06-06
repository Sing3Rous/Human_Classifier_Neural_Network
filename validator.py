import cv2
import tensorflow as tf
import os
from tqdm import tqdm
from pathlib import Path
from convolutional_neural_network import load_model

def make_image_data(dir, imageSize):
    path = os.path.join(dir + "\\" + "validation")
    data = []
    for image in tqdm(os.listdir(path)):
        imageArray = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
        normalizedImageArray = cv2.resize(imageArray, (imageSize, imageSize))
        data.append(normalizedImageArray.reshape(-1, imageSize, imageSize, 3))

    return data


def rename_each_image(dir, additionalNames=[], isInit=False, isLast=False):
    path = os.path.join(dir + "\\" + "validation")
    num = 1
    i = 0
    if (isInit):
        for image in tqdm(os.listdir(path)):
            os.rename(path + "\\" + image, path + "\\" + str(num) + "_")
            num += 1
    elif (not isLast):
        for image in tqdm(os.listdir(path)):
            os.rename(path + "\\" + image, path + "\\" + image + additionalNames[i] + "_")
            i += 1
    else:
        for image in tqdm(os.listdir(path)):
            os.rename(path + "\\" + image, path + "\\" + image + ".jpg")

def rename_by_prediction(dir, models, categories):
    path = os.path.join(dir + "\\" + "validation")
    data = make_image_data(dir, 200)
    predictions = []
    num = 0

    for model in models:
        predictions.clear()
        for image in data:
            prediction = model.predict(image)
            predictions.append(categories[num][int(prediction[0][0])])
        rename_each_image(dir, predictions)
        num += 1

dir = "C:\\Users\\singe\\Documents\\Human Classifier"

categories = []
genderCategories = ["male", "female"]
categories.append(genderCategories)
ageCategories = ["adult", "child"]
categories.append(ageCategories)
hairColorCategories = ["blonde", "dark"]
categories.append(hairColorCategories)

models = []
genderModel = load_model("gender_conv[3]_filters[32]_batches[8]", dir, "gender")
models.append(genderModel)
ageModel = load_model("age_conv[3]_filters[16]_batches[16]", dir, "age")
models.append(ageModel)
hairColorModel = load_model("hair_color_conv[3]_filters[32]_batches[32]", dir, "hair_color")
models.append(hairColorModel)

rename_each_image(dir, isInit=True)
rename_by_prediction(dir, models, categories)
rename_each_image(dir, isLast=True)


