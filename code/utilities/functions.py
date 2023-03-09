import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

from minepy import MINE
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector


def plt_params(dpi, width, height):
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.figsize'] = (width,height)


def mic_matrix(data):

    matrix = pd.DataFrame(columns=data.columns, index=data.columns)

    for x, y in combinations(data.columns, 2):

        # MIC (default params)
        mine = MINE() 
        mine.compute_score(data[x], data[y]) 
        matrix.loc[[x],[y]] = mine.mic()
        matrix.loc[[y],[x]] = mine.mic()

    for col in data.columns:
        # MIC (default params)
        mine = MINE() 
        mine.compute_score(data[col], data[col]) 
        matrix.loc[[col],[col]] = mine.mic()

    return matrix.apply(pd.to_numeric)


def transform(X, y, classes=None, shuffle=True, type_of_model='regressor'):

    # Create a transformation pipeline to scale the numerical variables and one hot encode the categorical. 
    Transformer = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(), make_column_selector(dtype_include=object)), remainder='passthrough')

    X = Transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=shuffle)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=shuffle)

    if not type_of_model == 'regressor':
        y_test = tf.one_hot(y_test, classes)
        y_train = tf.one_hot(y_train, classes)
        y_valid = tf.one_hot(y_valid, classes)

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def metrics_printer(metrics, machine='class', p=False):
    print('TEST METRICS')
    if p:
        if machine == 'class':
            (print('negloglike \t {:.2f} \ncatcrossent \t {:.2f} \nAccuracy \t {:.2f} \nPrecison \t {:.2f} \nRecall  \t {:.2f}'
                    .format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])))
        if machine == 'regressor':
            print('negloglike \t {:.2f} \nMSE \t\t {:.2f}'.format(metrics[0], metrics[1]))
    else:
        if machine == 'class':
            (print('catcrossent \t {:.2f} \nAccuracy \t {:.2f} \nPrecison \t {:.2f} \nRecall  \t {:.2f}'
                    .format(metrics[0], metrics[1], metrics[2], metrics[3])))
        if machine == 'regressor':
            print('MSE \t {:.2f}'.format(metrics))


def metrics_plotter(history, metrics, loss):

    histories = pd.DataFrame(history.history)

    n = 1 if isinstance(metrics, float) else len(metrics)
    
    fig, ax = plt.subplots(1, None if isinstance(metrics, float) else len(metrics), sharex=True)

    for i, m in enumerate(['loss'] if isinstance(metrics, float) else metrics):
        subset = histories.filter(regex=m).reset_index().melt(id_vars="index", var_name="subset")
        sns.lineplot(data=subset, x='index', y='value', hue='subset', lw=0.85, alpha=0.75, legend=False, ax=ax if isinstance(metrics, float) else ax[i])
        if m == 'loss': 
            m = loss
        if isinstance(metrics, float):
            ax.set_title(m, fontsize=10)
            ax.set_xticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax[i].set_title(m, fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")

    plt.tight_layout()
    plt.show()


def inspect_images(data, num_images):

    n = int(np.sqrt(num_images))
    fig, ax = plt.subplots(nrows=n, ncols=n)

    i = 0; j = 0 
    for img in range(num_images):
        subset = data.iloc[img, 1:data.shape[1]].to_numpy().reshape(28,28)
        ax[i,j].imshow(subset, cmap='gray')
        ax[i,j].axis('off')
        j+=1
        if j == n:
            i+=1
            j=0

    plt.tight_layout()
    plt.show()

