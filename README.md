# Human Classifier
### Convolutional neural network that classifies human characteristics by the image of its face.
##### Count, nature and classes of categories determines by you.

##### `make_pickles.py` - script that makes pickles from images.

##### `convolutional_neural_network.py` - script that builds and trains convolutional neural network models.

##### `validator.py` - script that uses prediction of some models (builded and trained by using `convolutional_neural_network.py`) and renames every image in folder by prediction (e.g. 10_male_adult_blonde).

##### `utkface_dataset_parser.py` - script for parsing [UTKFace dataset](https://susanqq.github.io/UTKFace/) .

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
json
datetime
shutil
~~~~

#### For correct work you must install your directories structure following this scheme:
![alt text](https://raw.githubusercontent.com/Sing3Rous/Human_Classifier_Neural_Network/master/directory%20tree.png)

#### where:

![alt text](https://raw.githubusercontent.com/Sing3Rous/Human_Classifier_Neural_Network/master/sets%20tree.png)

##### You must create each colorful field manually, but light grey fields are optional: scripts may create them for you.
##### Gradient fields in some cases would be created by scripts, but other cases you must create it by your own (which cases exactly determines by the first tree (light gray - manually, colorful - opposite)).

#### Documents on _russian_:

##### [Terms of reference (obsolete)](https://github.com/Sing3Rous/Human_Classifier_Neural_Network/blob/master/Terms%20Of%20Reference.pdf)

##### [Research report (obsolete)](https://github.com/Sing3Rous/Human_Classifier_Neural_Network/blob/master/Research%20Report.pdf)
