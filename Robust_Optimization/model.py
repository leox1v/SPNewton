import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
from newton import SPNewtonOptimizer, NewtonOptimizer, SimultNewtonOptimizer


class HSNet():
    def __init__(self, input_size=13, out_size=3, batch_size=100):
        self.input_size = input_size
        self.out = out_size
        self.batch_size = batch_size

        self.X = tf.placeholder("float", [None, self.input_size])
        self.Y = tf.placeholder("float", [None, self.out])

        self.X_test = self.X

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.input_size, 2], stddev=1.)),
            'h2': tf.Variable(tf.random_normal([2, 1], stddev=1.)),
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([2])),
            'b2': tf.Variable(tf.zeros([1]))
        }

        self.train, self.loss, self.grad = self.train_op()

        self.pred = self.test_op()

        self.test_loss = self.loss

    def logit(self, x):
        # logit_ = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        x1 = tf.nn.tanh(tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1']))
        logit_ = tf.add(tf.matmul(x1, self.weights['h2']), self.biases['b2'])
        return logit_

    def train_op(self):
        logits = self.logit(self.X)
        target = self.decode_one_hot(self.Y)

        # Logistic Loss
        thr_condition = tf.greater(target, 0.)
        pred = tf.nn.sigmoid(logits)
        loss = tf.where(thr_condition, -self.mylog(pred), -self.mylog(1-pred))
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        return optimizer.minimize(loss), loss, logits

    def test_op(self):
        logits = self.logit(self.X)
        pred = tf.nn.sigmoid(logits)
        return pred

    def decode_one_hot(self, x):
        x_new = tf.cast(tf.argmax(x, axis=1), tf.float32)
        thr_condition = tf.less(x_new, 0.5)
        x_new = tf.where(thr_condition, -tf.ones_like(x_new), tf.ones_like(x_new))
        return x_new

    def mylog(self, x):
        x = tf.clip_by_value(x, 1E-5, 1E100)
        return tf.log(x)



class HSRobustNet(HSNet):
    def __init__(self, input_size=13, out_size=3, batch_size=100, optim='Adam'):
        self.input_size = input_size
        self.out = out_size
        self.optim = optim
        self.batch_size = batch_size
        self.flipping = len(optim.split(" ")) > 1

        self.X = tf.placeholder("float", [batch_size, self.input_size])
        self.Y = tf.placeholder("float", [None, self.out])

        self.X_test = tf.placeholder("float", [None, self.input_size])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.input_size, 2], stddev=1.)),
            'h2': tf.Variable(tf.random_normal([2, 1], stddev=1.))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([2])), 'b2': tf.Variable(tf.zeros([1]))
        }

        self.p = tf.Variable(tf.divide(tf.ones([1, batch_size], tf.float32), batch_size))

        self.p_normalized = np.divide(self.p, tf.reduce_sum(self.p))

        self.theta = [self.weights['h1'], self.weights['h2'], self.biases['b1'], self.biases['b2']]
        self.phi = [self.p]

        self.train, self.loss = self.train_op()
        if self.flipping:
            (self.train_1, self.train_2) = self.train
        self.pred, self.test_loss = self.test_op()

        self.chi2 = self.chi2_dist(self.p_normalized)

        self.grad_norm = self.gradient_norm_op(self.loss)

        self.perturb_theta = self.perturb_params(self.theta)
        self.perturb_phi = self.perturb_params(self.phi)

    def gradient_norm_op(self, loss):
        grad1 = tf.gradients(loss, self.theta)
        grad1 = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in grad1], axis=0)
        grad1 = tf.reduce_sum(tf.square(grad1))

        grad2 = tf.gradients(-loss, self.phi)
        grad2 = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in grad2], axis=0)
        grad2 = tf.reduce_sum(tf.square(grad2))
        return (grad1, grad2)

    def train_op(self):
        logits = self.logit(self.X)
        target = self.decode_one_hot(self.Y)

        # Logistic Loss
        thr_condition = tf.greater(target, 0.)
        pred = tf.nn.sigmoid(logits)
        loss_1 = tf.where(thr_condition, -self.mylog(pred), -self.mylog(1 - pred))

        p_abs = tf.maximum(tf.zeros_like(self.p), self.p)
        loss_1 = tf.matmul(p_abs, loss_1)

        reg = tf.reduce_sum(tf.square(p_abs - tf.ones_like(p_abs) * tf.divide(1., self.batch_size)))

        loss = loss_1 - 1. * reg

        # loss2 = loss_1 - 1. * reg

        if self.flipping:
            optimizer_1 = self.optimize_op(self.optim.split(" ")[0], loss)
            optimizer_2 = self.optimize_op(self.optim.split(" ")[1], loss)
            optimizer = (optimizer_1, optimizer_2)
        else:
            optimizer = self.optimize_op(self.optim, loss)
        return optimizer, loss

    def optimize_op(self, optim, loss):
        if optim == "Adam":
            optim_theta = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, var_list=self.theta)
            optim_phi = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-loss, var_list=self.phi)
            optimizer = (optim_theta, optim_phi)
        elif optim == "GD":
            optim_theta = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss, var_list=self.theta)
            optim_phi = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(-loss, var_list=self.phi)
            optimizer = (optim_theta, optim_phi)
        elif optim == "SPNewton":
            newton_opt = SPNewtonOptimizer(self.theta, self.phi, 0.01)
            optimizer, _ = newton_opt.step(loss)
        elif optim == "ASPNewton":
            newton_opt = AdditiveSPNewtoonOptimizer(self.theta, self.phi, 1.)
            optimizer, _ = newton_opt.step(loss, -loss)
        elif optim == "Newton":
            newton_opt = NewtonOptimizer(self.theta, self.phi, 0.01)
            optimizer, _ = newton_opt.step(loss)
        elif optim == "SimultNewton":
            newton_opt = SimultNewtonOptimizer(self.theta, self.phi, 0.01)
            optimizer, _ = newton_opt.simult_step(loss, -loss)
        else:
            raise NotImplementedError
        return optimizer

    def test_op(self):
        logits = self.logit(self.X_test)
        pred = tf.nn.sigmoid(logits)
        target = self.decode_one_hot(self.Y)

        # Logistic Loss
        thr_condition = tf.greater(target, 0.)
        loss = tf.reduce_mean(tf.where(thr_condition, -self.mylog(pred), -self.mylog(1 - pred)))

        return pred, loss

    def perturb_params(self, param_list):
        assign_ops = []
        for param in param_list:
            perturb_tensor = tf.random_normal(param.shape, mean=0.0, stddev=.1)
            assign_ops.append(tf.assign_add(param, perturb_tensor))
        return assign_ops


    def chi2_dist(self, p):
        # dist = tf.reduce_sum(tf.square(p * self.batch_size - 1)) * tf.divide(1.0, 2* self.batch_size)
        p = tf.abs(p)
        dist = tf.reduce_mean(tf.square(tf.multiply(p, self.batch_size) - tf.ones_like(p)))
        return dist


    def reset(self):
        tf.reset_default_graph()