import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def B1():
    train_root="./Dataset/dataset_AMLS_22-23/cartoon_set"
    test_root="./Dataset/dataset_AMLS_22-23_test/cartoon_set_test"

    dataframe = pd.read_csv(f"{train_root}/labels.csv")
    Y_train = dataframe["face_shape"].to_numpy().reshape(-1, 1)
    X_train = [np.asarray(Image.open(f"{train_root}/img/{dataframe['file_name'][j]}").convert("RGB").crop((100, 50, Image.open(f"{train_root}/img/{dataframe['file_name'][j]}").size[0]-100, Image.open(f"{train_root}/img/{dataframe['file_name'][j]}").size[1]-50)).resize((60, 80))).flatten() for j in range(10000)]
    X_train = np.array(X_train)

    test_dataframe = pd.read_csv(f"{test_root}/labels.csv")
    Y_test = test_dataframe["face_shape"].to_numpy().reshape(-1, 1)
    X_test = [np.asarray(Image.open(f"{test_root}/img/{test_dataframe['file_name'][i]}").convert("RGB").crop((100, 50, Image.open(f"{test_root}/img/{test_dataframe['file_name'][i]}").size[0]-100, Image.open(f"{test_root}/img/{test_dataframe['file_name'][i]}").size[1]-50)).resize((60, 80))).flatten() for i in range(2500)]
    X_test = np.array(X_test)

    result_RF = RF(X_train, Y_train, X_test, Y_test)


def RF(X_train, Y_train, X_test, Y_test):
    print('starting RF')
    rf = RandomForestClassifier(n_estimators=11, random_state=5, verbose=1)
    parameter_grid = {'max_depth' : np.arange(1,20,1)}
    grid_search = GridSearchCV(rf, parameter_grid, cv=5)
    grid_search.fit(X_train, Y_train.ravel())
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    x1, y1 = [], []
    results = grid_search.cv_results_
    test_accuracy = results['mean_test_score']
    test_params = results['params']
    for mean, param in zip(test_accuracy, test_params):
        y1.append(mean)
        x1.append(int(param['max_depth']))

    Y_train = Y_train.ravel()
    rf_classifier = RandomForestClassifier(max_depth=14)
    rf_classifier.fit(X_train, Y_train)
    Y_pred = rf_classifier.predict(X_test)

    print('accuracy_score: ', accuracy_score(Y_test, Y_pred))
    print('confusion_matrix:\n ', confusion_matrix(Y_test, Y_pred))

    plot_results(x1, y1, "test_accuracy vs. max_depth", 
                 "max_depth", "test_accuracy", "test_accuracy_vs_max_depth_plot.jpg")

    return rf_classifier, best_params, best_score


def plot_results(x, y, title, x_label, y_label, filename):
    plt.plot(x, y, 'o-')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)

B1()