#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import graphviz


def clean_data(df):
    # drop unnecessary columns
    frames = [df['fire_size'], df['fire_size_class'],
              df.loc[:, 'Temp_pre_30':'Prec_cont']]
    df = pd.concat(frames, axis=1)

    # specify and drop the missing data
    df = df.replace(0, np.nan)
    df = df.replace(-1, np.nan)
    df = df.dropna().reset_index()

    # select data for classification
    data = df.loc[:, 'Temp_pre_30':'Prec_cont']
    data = data.apply(lambda x: pd.qcut(x, 7, labels=list(range(7))), axis=0)
    frames = [df['fire_size_class'], data]
    class_data = pd.concat(frames, axis=1)

    # select data for regression
    frames = [df['fire_size'], df.loc[:, 'Temp_pre_30':'Prec_cont']]
    re_data = df = pd.concat(frames, axis=1)

    return (class_data,  re_data)


def classification(df, depth):
    # create categorical features
    features = df.loc[:, df.columns != 'fire_size_class']
    features = pd.get_dummies(features)
    labels = df['fire_size_class']

    # split data into training and testing group
    features_train, features_test, labels_train, labels_test =         train_test_split(features, labels, test_size=0.2)

    # train and test model by DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(features_train, labels_train)
    test_predictions = model.predict(features_test)
    print('Result from Classification at MAX_DEPTH = ' + str(depth))
    print('    Test  Accuracy:', accuracy_score(labels_test, test_predictions))
    print()
    

def regression(df):
    # preview correlation between vairables
    fig_1, ax1 = plt.subplots(1, figsize=(13, 10))
    corr = df.corr()
    sns.heatmap(ax=ax1, data=corr, xticklabels=corr.columns,
                yticklabels=corr.columns, fmt='.4f', annot=False,
                vmin=-1, vmax=1, center=0, cmap='coolwarm')
    plt.xticks(rotation=-45)
    ax1.set_title('Possible Correlation Map')
    fig_1.savefig('correlation.png')

    # split data into training and testing group
    X = df.loc[:, df.columns != 'fire_size']
    Y = df['fire_size']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33,
                                                        shuffle=True)

    # test model by linear regression
    lineReg = LinearRegression()
    lineReg.fit(X_train, y_train)
    print('Result from Linear Regression:')
    print('    Score: ', lineReg.score(X_test, y_test))
    print('    Weights: ', lineReg.coef_)
    print()


def main():
    # clean data
    raw = pd.read_csv('FW_Veg_Rem_Combined.csv')
    class_data, re_data = clean_data(raw)

    # construct predictive model by classification
    classification(class_data, 4)

    # construct predictive model by regression
    regression(re_data)


if __name__ == '__main__':
    main()


# In[ ]:




