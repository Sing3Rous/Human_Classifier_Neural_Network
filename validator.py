import cv2
import os
from pathlib import Path
from tqdm import tqdm
from util import load_model

def main():
    dir = "C:\\Users\\singe\\Documents\\Human Classifier"

    classes = get_classes(dir)
    models = get_models(dir)

    rename_each_image(dir, isInit=True)
    rename_by_prediction(dir, models, classes)
    rename_each_image(dir, isLast=True)

#takes each image in folder and makes an data array of every image
#returns an array of arrays of data of images
def make_image_data(dir, imageSize):
    path = os.path.join(dir, "validation", "images")
    data = []
    for image in tqdm(os.listdir(path)):
        imageArray = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
        normalizedImageArray = cv2.resize(imageArray, (imageSize, imageSize))
        data.append(normalizedImageArray.reshape(-1, imageSize, imageSize, 3))

    return data

#takes each image in folder and renames it:
#1) if its an initialization: renames to 1_, 2_, 3_, ..., n_
#2) if its last iteration: renames to %name%.jpg
#3) else: renames to %name%%additional name%_, where additional name is
#a name in a list of strings
def rename_each_image(dir, additionalNames=[], isInit=False, isLast=False):
    path = os.path.join(dir, "validation", "images")
    num = 1
    if (isInit):
        for image in tqdm(os.listdir(path)):
            os.rename(os.path.join(path, image), os.path.join(path, str(num) + "_"))
            num += 1
    elif (isLast):
        for image in tqdm(os.listdir(path)):
            os.rename(os.path.join(path, image), os.path.join(path, image + ".jpg"))
    else:
        for image in tqdm(os.listdir(path)):
            os.rename(os.path.join(path, image), os.path.join(path, image + additionalNames[num - 1] + "_"))
            num += 1

#applies prediction of each model on each image in folder
#and renames it
def rename_by_prediction(dir, models, categories):
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

#returns list of lists of classes of each category
def get_classes(dir):
    classes = []
    path = os.path.join(dir, "categories")
    for category in os.listdir(path):
        categoryClasses = os.listdir(os.path.join(path, category, "train"))
        classes.append(categoryClasses)

    return classes

#returns list of all models in ../validation/models folder
def get_models(dir):
    models = []
    path = os.path.join(dir, "validation", "models")
    for modelName in os.listdir(path):
        model = load_model(modelName, dir)
        models.append(model)

    return models

if __name__ == "__main__":
    main()

