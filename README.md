# Human Classifier
### Convolutional neural network that classifies some of human characteristics by the image of his face.
##### This cnn based on such classes (binaric) as: gender, age, hair color.

##### `make_pickles.py` - script that makes pickles from images.

##### `convolutional_neural_network.py` - script that builds and trains convolutional neural network models.

##### `validator.py` - script that uses prediction of some models (builded and trained by using `convolutional_neural_network.py`) and renames every image in folder by prediction (e.g. 10_male_adult_blonde).

#### Used libraries:
~~~~
tensorflow
tensorflow.keras
pickle
time
os
numpy
cv2
tqdm
random
~~~~

### For correct work you must install your directories structure following this scheme:
![alt text](https://raw.githubusercontent.com/Sing3Rous/Human_Classifier_Neural_Network/master/directory%20tree.png)

### where:

![alt text](https://raw.githubusercontent.com/Sing3Rous/Human_Classifier_Neural_Network/master/sets%20tree.png)

#### You must create each colorful field manually, but light grey fields are optional: scripts may create them for you.

#### Documents on _russian_:

##### [Terms of reference](https://github.com/Sing3Rous/Human_Classifier_Neural_Network/blob/master/Terms%20Of%20Reference.pdf)

##### [Research report](https://github.com/Sing3Rous/Human_Classifier_Neural_Network/blob/master/Research%20Report.pdf)
