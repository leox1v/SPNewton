import tensorflow as tf
import numpy as np
import os
from scipy.stats import multivariate_normal
import pickle

def sample_Z(m, n):
    '''Gaussian prior for G(Z)'''
    return np.random.normal(size=[m,n])

def setup_directories(*dirs):
    for _dir in dirs:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

def get_flags(dataset, exp_no):
    path = "results/{}/exp_{}/checkpoint/flags".format(dataset, exp_no)
    with open(path, 'rb') as file:
        FLAGS = pickle.load(file)
    return FLAGS

class Helper():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.exp_dir = None
        self.exp_no = 0

    def setup_directories(self):
        result_dir = "results/" + self.FLAGS.dataset + "/"
        setup_directories(result_dir)

        i = 0
        self.exp_dir = result_dir + "exp_{}/".format(i)
        while os.path.exists(self.exp_dir):
            i += 1
            self.exp_dir = result_dir + "exp_{}/".format(i)
        self.exp_no = i
        os.makedirs(self.exp_dir)

        self.FLAGS.output_dir = self.exp_dir + self.FLAGS.output_dir
        self.FLAGS.summaries_dir = self.exp_dir + self.FLAGS.summaries_dir
        self.FLAGS.array_dir = self.exp_dir + self.FLAGS.array_dir
        self.FLAGS.checkpoint_dir = self.exp_dir + self.FLAGS.checkpoint_dir

        setup_directories(self.FLAGS.output_dir, self.FLAGS.summaries_dir, self.FLAGS.array_dir, self.FLAGS.checkpoint_dir)

        with open(self.FLAGS.checkpoint_dir + "flags", 'wb') as file:
            pickle.dump(self.FLAGS, file, protocol=pickle.HIGHEST_PROTOCOL)

        return self.FLAGS



def setup_eigval_dict(_dict, optimizer):
    _dict[optimizer] = []
    _dict['{}_ul'.format(optimizer)] = []
    _dict['{}_lr'.format(optimizer)] = []
    return _dict

def update_eigval_dict(_dict, optimizer, eigv, eigv_ul, eigv_lr):
    _dict[optimizer].append(eigv)
    _dict['{}_ul'.format(optimizer)].append(eigv_ul)
    _dict['{}_lr'.format(optimizer)].append(eigv_lr)
    return _dict



def get_sample_pdf(means, variances, cat_probs, modes, X):
    sums = np.zeros(np.array(X).shape[0])
    for n, x in enumerate(X):
        for i in range(modes):
            sums[n] += cat_probs[i] * multivariate_normal.pdf(x, means[i], variances[i])
    return sums


def load_opt_arrays(FLAGS):
    _dir = FLAGS.array_dir
    opt_methods_keys = FLAGS.opt_methods.split(" ")
    opt_methods = dict()
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    for method in opt_methods_keys:
        opt_methods[method + "_g"] = np.array([])
        opt_methods[method + "_d"] = np.array([])
        if os.path.isfile(_dir + method +"_g.npy"):
            opt_methods[method + "_g"] = np.load(_dir + method +"_g.npy")
        if os.path.isfile(_dir + method +"_d.npy"):
            opt_methods[method + "_d"] = np.load(_dir + method +"_d.npy")

    return opt_methods

def save_opt_methods(opt_methods, out_dir):
    for key in opt_methods.keys():
        np.save(out_dir + key + ".npy", opt_methods[key])

def set_seed(seed=None):
    if seed is None:
        seed = np.random.choice(150)
    print("Random Seed {}".format(seed))
    np.random.seed(seed)
    tf.set_random_seed(seed)
    return seed

def safe_append(dict, key, val):
    if key not in dict.keys():
        dict[key] = list([val])
    else:
        dict[key].append(val)
    return dict

def save_log(x):
    x = np.clip(x, 1E-10, 1E10)
    return np.log(x)