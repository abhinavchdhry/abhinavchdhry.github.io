---
layout: post
title:  "Fish classification using Keras"
tagline: "Practical intro to deep learning using Keras"
date:   2017-04-15 20:20:57 -0400
categories: jekyll update
---


Keras is a powerful framework for building deep learning models that facilitates rapid prototyping. It works on top of a tensor manipulation engine (either Tensorflow or Theano) which are optimized to leverage the GPU for fast, parallel matrix computations.

### Data organization and preprocessing
For an image classification task, you are likely to find your image dataset in a number of different formats:
1. A folder containing all the training images and a separate text file designating the class ids for each image
2. Images named by concatenating their respective classes with an unique identifier eg. dog_001.jpg, cat_002.jpg, etc
3. Separate directories for each class containing images of that class

The image preprocessing library in Keras expects data to be in the third format. These inbuilt image reading and batching functionalities in Keras makes life a lot easier by allowing you to

### Creating training and validation sets
We need to split the data into a training set which will be used to train our deep learning model, and a validation set which would allow us to test how well our model classifies instances it has not seen before. This is a measure of the **generalization capability** of the model, and is crucial in reducing overfitting. The ratio of the split is usually set at 70:30 or 80:20.

The idea is to use a statistical sampling technique called the **Stratified Split**. If the dataset is split randomly and there are only a few of images for certain classes, it is likely that either the training set or the validation set will end up with no instances of those classes. When the model observes no instances of a certain class in its training data, its classification performance on instances of that class will likely be poor. Stratified split solves this problem by sampling instances from each class based on the set split ratio, thus ensuring both the training and validation sets have instances of every class.

Below is a function that uses Scikit-learn's [StratifiedShuffleSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) function to create a 80:20 split

```python
from sklearn.model_selection import StratifiedShuffleSplit

def stratifiedSplit(X, Y, test_size=0.2):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=2017)
        sss.get_n_splits(X, Y)
        for train_idx, test_idx in sss.split(X, Y):
                break
        return(train_idx, test_idx)
```

Next, we use the function to create training and validation datasets:

```python

ORIGINAL_DATAPATH = "./train/"
TRAIN_PATH = "./TRAIN/"
VALIDATION_PATH = "./VALID/"

def createTrainAndValidationDatasets(datapath):
        print("### Sampling training and validation datasets...")
        classes = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]

        def makeDirectoryStructure(path):
                if not os.path.exists(path):
                        for cl in classes:
                                os.makedirs(path + cl)
                else:
                        shutil.rmtree(path)
                        for cl in classes:
                                os.makedirs(path + cl)

        img_names = []
        img_classes = []
        for cl in classes:
                class_dir = ORIGINAL_DATAPATH + cl + "/"
                filepaths = glob.glob(class_dir + "*.jpg")
                for filepath in filepaths:
                        img_names.append(os.path.basename(filepath))
                        img_classes.append(cl)

        train_idx, valid_idx = stratifiedSplit(img_names, img_classes)
        makeDirectoryStructure(TRAIN_PATH)
        makeDirectoryStructure(VALIDATION_PATH)

        for idx in train_idx:
                img_name = img_names[idx]
                img_class = img_classes[idx]
                shutil.copyfile(ORIGINAL_DATAPATH + img_class + "/" + img_name,\
                	TRAIN_PATH + img_class + "/" + img_name)

        for idx in valid_idx:
                img_name = img_names[idx]
                img_class = img_classes[idx]
                shutil.copyfile(ORIGINAL_DATAPATH + img_class + "/" + img_name,
               		VALIDATION_PATH + img_class + "/" + img_name)

createTrainAndValidationDatasets(ORIGINAL_DATAPATH)
```

