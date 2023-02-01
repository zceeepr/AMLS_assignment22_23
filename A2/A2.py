#rewritten imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import os
import lab2_landmarks as l2

from keras.preprocessing import image
from keras.utils import load_img, img_to_array

from sklearn import datasets, svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import tree

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,precision_score, recall_score, f1_score


def A2():
    train_root="./Dataset/dataset_AMLS_22-23/celeba"
    test_root="./Dataset/dataset_AMLS_22-23_test/celeba_test"


    X_train, Y_train = l2.extract_features_labels(os.path.join(train_root, "img"), train_root, "smiling")
    X_test, Y_test = l2.extract_features_labels(os.path.join(test_root, "img"), test_root, "smiling")

    X_train = np.array(X_train).reshape(len(X_train), -1)
    print("Number of X_train instances not captured:", 5000 - len(X_train))

    Y_train = np.array(Y_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(len(X_test), -1)
    Y_test = np.array(Y_test).reshape(-1, 1)

    result_SVM = SVM(X_train, Y_train, X_test, Y_test)


def SVM(X_train, Y_train, X_test, Y_test):
    print("\n" + 'SVM HYPERPARAMETER TUNING')
    

    hyperparameters = {'degree': [1], 
                  'C': [0.1, 1, 10, 100, 1000], 
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    grid_search = GridSearchCV(SVC(), hyperparameters, refit=True, verbose=3)
    grid_search.fit(X_train, Y_train.ravel())
    

    print('The best parameters and estimator:', grid_search.best_params_, grid_search.best_estimator_)

    FINAL_GRID = grid_search.predict(X_test)


    print("Final results for SVM model\n" +
      "Accuracy: {:.4f}\n".format(accuracy_score(Y_test, FINAL_GRID)) +
      "Confusion Matrix:\n" + str(confusion_matrix(Y_test, FINAL_GRID)) + "\n" +
      classification_report(Y_test, FINAL_GRID) + "\n" +
      "Precision: {:.4f}\n".format(precision_score(Y_test, FINAL_GRID)) +
      "Recall: {:.4f}\n".format(recall_score(Y_test, FINAL_GRID)) +
      "F1: {:.4f}".format(f1_score(Y_test, FINAL_GRID)))


A2()