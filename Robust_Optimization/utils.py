import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import tensorflow as tf

class Visualizer():
    def vanilla_plot(self, _dict, x_every=100, value="Accuracy", flip=False, log=False):
        # save_dict(_dict, value=value)
        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        offset = 0
        for i, key in enumerate(_dict.keys()):
            x_range = range(offset, offset + x_every*len(_dict[key]), x_every)
            if log:
                _dict[key] = save_log(_dict[key])
            plt.plot(x_range, _dict[key], label=key.replace("_"," cont'd"))
            if flip and i == 0:
                offset += x_every*(len(_dict[key]) - 1)
                offset_y = _dict[key][-1]

        plt.axvline(x=offset, linewidth=1, color='k', label="Perturbation")

        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(value)

        # plt.savefig(save_name("plots/plot_{}.png".format(value.replace(" ", "_"))))

        plt.show()
        plt.close()

    def acc_conf_plot(self, _dic, x_every=100, value="Accuracy"):
        sns.set()
        plt.figure()
        df = self.dict_to_pandas(_dic, x_every, value)
        sns.tsplot(data=df, time="Epoch", value=value, condition="Optimizer", unit="Runs", ci=90)
        # plt.savefig(save_name("plots/plot_{}.png".format(value.replace(" ", "_"))))

    def dict_to_pandas(self, _dict, x_every, value):
        columns = ["Runs", "Optimizer", value, "Epoch"]
        df = pd.DataFrame(columns=columns)
        for key in _dict.keys():
            arr = np.array(_dict[key])
            times = arr.shape[0]
            for time in range(times):
                for epoch, acc in enumerate(arr[time, :]):
                    df_new = pd.DataFrame([[time, key, acc, epoch*x_every]], columns=columns)
                    df = df.append(df_new)
        return df

class DataSet():
    def __init__(self, seed):
        self.seed = seed
        self.data = self.preprocess(load_breast_cancer())
        # self.data.data = preprocessing.scale(self.data.data)
        self.x_train, self.x_test, self.y_train, self.y_test = self.split()
        self.dim = self.x_train.shape[1]
        self.classes = len(np.unique(self.y_train))
        self.N_train = len(self.y_train)

    def preprocess(self, data, n_total=100):
        ind = np.random.choice(np.arange(len(data.target)), n_total, replace=False)
        data.data = preprocessing.scale(data.data)
        data.data = data.data[ind, :]
        data.target = data.target[ind]
        return data

    def split(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data.data, self.data.target, test_size=0.25, random_state=self.seed)
        return x_train, x_test, y_train, y_test

    def reshuffle_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = self.split()

def save_dict(_dict, value="acc"):
    with open(save_name('pickles/{}.pickle'.format(value.replace(" ", "_"))), 'wb') as handle:
        pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(fname):
    with open('pickles/{}.pickle'.format(fname), 'rb') as handle:
        return pickle.load(handle)

def save_name(fname='pickles/acc.pickle'):
    ext = fname.split(".")[-1]
    pre = fname.split(".")[0]
    i = 1
    while os.path.exists(fname):
        fname = "{}_{}.{}".format(pre, i, ext)
        i += 1

    return fname

def save_log(x):
    x = np.clip(x, 1E-10, 1E10)
    return np.log(x)

def set_seed(seed=None, verbose=False):
    if seed is None:
        seed = np.random.choice(150)
    if verbose:
        print("random Seed {}".format(seed))
    np.random.seed(seed)
    tf.set_random_seed(seed)

    return seed

def safe_append(dict, key, val):
    if key not in dict.keys():
        dict[key] = list([val])
    else:
        dict[key].append(val)
    return dict


# acc = load_dict("acc_flip_3")
# grad = load_dict("grad_flip_22")
# Visualizer().vanilla_plot(acc, flip=True)
# Visualizer().vanilla_plot(grad, value="Squared Gradientsum", flip=True)
# Visualizer().vanilla_plot(grad, value="Squared Gradientsum", flip=True, log=True)


# acc = load_dict("acc_3")
# grad = load_dict("grad_3")

# Visualizer().acc_conf_plot(acc, x_every=100, value="Accuracy")
# Visualizer().acc_conf_plot(grad, x_every=100, value="Squared Gradientsum")













