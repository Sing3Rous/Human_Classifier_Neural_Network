import os
from shutil import copy2
from tqdm import tqdm

class Person(object):
    def __init__(self, age, gender, race):
        self.age = age
        self.gender = gender
        self.race = race

    def get_age_group(self):
        if (self.age >= 56):
            return 3
        if (self.age < 56 and self.age >= 31):
            return 2
        if (self.age < 31 and self.age >= 16):
            return 1
        if (self.age < 16):
            return 0

def main():
    dir = "C:\\Users\\singe\\Documents\\UTKFace dataset"
    persons = parse(dir)
    parse_by_age_group(dir, persons)
    parse_by_gender(dir, persons)
    parse_by_race(dir, persons)

def parse(dir):
    path = os.path.join(dir, "images")
    persons = []
    for image in os.listdir(path):
        number = ""
        underscoresCount = 0
        age = -1
        gender = -1
        race = -1
        for symbol in image:
            if (symbol != "_"):
                number += symbol
            else:
                underscoresCount += 1

                #age group detection
                if (underscoresCount == 1):
                    age = int(number)
                    number = ""

                #gender detection
                if (underscoresCount == 2):
                    gender = int(number)
                    number = ""

                #race detection
                #and finishing parsing image name
                if (underscoresCount == 3):
                    race = int(number)
                    persons.append(Person(age, gender, race))
                    break

    return persons

def parse_by_age_group(dir, persons):
    imagesPath = os.path.join(dir, "images")
    path = os.path.join(dir, "age")
    if (not os.path.exists(path)):
        os.mkdir(path)
    groupZeroPath = os.path.join(path, "0")
    groupOnePath = os.path.join(path, "1")
    groupTwoPath = os.path.join(path, "2")
    groupThreePath = os.path.join(path, "3")
    if (not os.path.exists(groupZeroPath)):
        os.mkdir(groupZeroPath)
    if (not os.path.exists(groupOnePath)):
        os.mkdir(groupOnePath)
    if (not os.path.exists(groupTwoPath)):
        os.mkdir(groupTwoPath)
    if (not os.path.exists(groupThreePath)):
        os.mkdir(groupThreePath)

    imagesFullPath = [os.path.join(imagesPath, image) for image in os.listdir(imagesPath)]
    for image, person in tqdm(zip(imagesFullPath, persons), total=len(persons)):
        ageGroup = person.get_age_group()
        if (ageGroup == 0):
            copy2(image, groupZeroPath)
        if (ageGroup == 1):
            copy2(image, groupOnePath)
        if (ageGroup == 2):
            copy2(image, groupTwoPath)
        if (ageGroup == 3):
            copy2(image, groupThreePath)

def parse_by_gender(dir, persons):
    imagesPath = os.path.join(dir, "images")
    path = os.path.join(dir, "gender")
    if (not os.path.exists(path)):
        os.mkdir(path)
    malePath = os.path.join(path, "male")
    femalePath = os.path.join(path, "female")
    if (not os.path.exists(malePath)):
        os.mkdir(malePath)
    if (not os.path.exists(femalePath)):
        os.mkdir(femalePath)

    imagesFullPath = [os.path.join(imagesPath, image) for image in os.listdir(imagesPath)]
    for image, person in tqdm(zip(imagesFullPath, persons), total=len(persons)):
        if (person.gender == 0):
            copy2(image, malePath)
        if (person.gender == 1):
            copy2(image, femalePath)

def parse_by_race(dir, persons):
    imagesPath = os.path.join(dir, "images")
    path = os.path.join(dir, "race")
    if (not os.path.exists(path)):
        os.mkdir(path)
    europidPath = os.path.join(path, "europid")
    negroidPath = os.path.join(path, "negroid")
    mongoloidPath = os.path.join(path, "mongoloid")
    dravidianPath = os.path.join(path, "dravidian")
    otherPath = os.path.join(path, "other")
    if (not os.path.exists(europidPath)):
        os.mkdir(europidPath)
    if (not os.path.exists(negroidPath)):
        os.mkdir(negroidPath)
    if (not os.path.exists(mongoloidPath)):
        os.mkdir(mongoloidPath)
    if (not os.path.exists(dravidianPath)):
        os.mkdir(dravidianPath)
    if (not os.path.exists(otherPath)):
        os.mkdir(otherPath)

    imagesFullPath = [os.path.join(imagesPath, image) for image in os.listdir(imagesPath)]
    for image, person in tqdm(zip(imagesFullPath, persons), total=len(persons)):
        if (person.race == 0):
            copy2(image, europidPath)
        if (person.race == 1):
            copy2(image, negroidPath)
        if (person.race == 2):
            copy2(image, mongoloidPath)
        if (person.race == 3):
            copy2(image, dravidianPath)
        if (person.race == 4):
            copy2(image, otherPath)

if __name__ == "__main__":
    main()