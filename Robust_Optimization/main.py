import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from utils import *
from logistic_regression import LogisticNet, LogisticRobustNet
from model import HSNet, HSRobustNet
from sklearn import preprocessing


class Trainer():
    def __init__(self, model, imgs, target, imgs_test, target_test, batch_size=100, full_batch=False):
        self.imgs = imgs
        self.target = target
        self.full_batch = full_batch
        if not full_batch:
            self.batch_size = batch_size
        else:
            self.batch_size = len(target)
        self.model = model
        self.imgs_test = imgs_test
        self.target_test = target_test
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, epochs=1, test_every=100, train_fun=None, grad_thr=0.):
        if train_fun is None:
            train_fun = self.model.train
        N = len(self.target)
        acc = []
        grad = []
        thr_counter = 0
        for epoch in range(epochs):
            avg_loss = 0
            total_batch = N // self.batch_size
            for i in range(total_batch):
                batch_x = self.imgs[i * self.batch_size:(i + 1) * self.batch_size, :]
                batch_y = self.one_hot_encoding(self.target[i * self.batch_size:(i + 1) * self.batch_size])

                _, loss = self.sess.run([train_fun, self.model.loss],
                                        {self.model.X: batch_x, self.model.Y: batch_y})
                avg_loss += loss / float(total_batch)


            if epoch % test_every == 0:
                chi2dist, grad_norm = self.sess.run([self.model.chi2, self.model.grad_norm], {self.model.X: batch_x, self.model.Y: batch_y})
                grad.append(np.sum(grad_norm))

                pred, loss = self.test()
                acc_ = np.mean(np.equal(pred, self.target_test))
                acc.append(acc_)

                print("Epoch: {}, chi2dist: {}, Grad_norm: {}, Accuracy: {}".format(epoch, chi2dist, grad_norm, acc_))

                if np.sum(grad_norm) < grad_thr:
                    thr_counter += 1
                    # If the gradient norm is smaller than the threshold for 3 consecutive times, stop the procedure.
                    if thr_counter > 2:
                        print("Stopped after Epoch {}".format(epoch))
                        break
                else:
                    thr_counter = 0


        return np.array(acc), np.array(grad)

    def test(self, verbose=False):
        batch_x = self.imgs_test
        batch_y = self.one_hot_encoding(self.target_test)
        pred, loss = self.sess.run([self.model.pred, self.model.test_loss],
                                   {self.model.X_test: batch_x, self.model.Y: batch_y})
        pred = np.reshape(np.around(pred, 0), -1)
        acc = np.mean(np.equal(pred, self.target_test))
        if verbose:
            print("Accuracy: {}, Test loss: {}".format(acc, loss))

            if acc < 0.6:
                p = self.sess.run(self.model.p_normalized)
                print(p)

        return pred, loss

    def perturb(self, _var="theta"):
        if _var == 'theta':
            self.sess.run(self.model.perturb_theta)
        elif _var == 'phi':
            self.sess.run(self.model.perturb_phi)


    def one_hot_encoding(self, y):
        n = len(np.unique(self.target))
        y_oh = np.zeros((len(y), n))
        y_oh[np.arange(len(y)), y] = 1
        return y_oh

    def decode_oh(self, y_oh):
        return np.argmax(y_oh, axis=1)


def main():
    train(epochs=2000, reps=10)
    # single_train_run()
    # flip_training()

def train(epochs, reps):
    dset = DataSet(np.random.randint(100))
    visualizer = Visualizer()
    acc = {}
    grad = {}
    models = {}

    models["SimultNewton"], models["Newton"], models["SPNewton"] = build_graphs(dset)
    # models["GD"], models["Adam"] = build_fast_graphs(dset)

    seeds = np.random.choice(100, reps, replace=False)

    for i,rep in enumerate(range(reps)):
        dset.reshuffle_split()
        for key in models.keys():
            set_seed(seeds[i])
            acc_, grad_ = Trainer(models[key], dset.x_train, dset.y_train, dset.x_test, dset.y_test, batch_size=dset.N_train).train(epochs, test_every=100)
            acc, grad = safe_append(acc, key, acc_), safe_append(grad, key, grad_)

    save_dict(acc, value="acc")
    save_dict(grad, value="grad")
    visualizer.acc_conf_plot(acc, x_every=100, value="Accuracy")
    visualizer.acc_conf_plot(grad, x_every=100, value="Squared Gradientsum")

def flip_training(optim="SimultNewton SPNewton", epochs=1500):
    seed = set_seed(13)
    dset = DataSet(seed)
    visualizer = Visualizer()
    acc = {}
    grad = {}
    # grad_thr = [0.0001, 0.000001]
    # grad_thr = [.0001, 0.000001]
    grad_thr = [0., 0.]
    epochs = [1500, 2500]

    model = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes,
                                       optim=optim)
    trainer = Trainer(model, dset.x_train, dset.y_train, dset.x_test, dset.y_test, batch_size=dset.N_train)
    for i, opt in enumerate(optim.split(" ")):
        train_fun = getattr(trainer.model, "train_{}".format(i + 1))
        acc[opt], grad[opt] = trainer.train(epochs=epochs[i], test_every=100, train_fun=train_fun, grad_thr=grad_thr[i])
        # if i == 1:
        #     acc[opt] = np.insert(acc[opt], 0, acc[optim.split(" ")[0]][-1])
        #     grad[opt] = np.insert(grad[opt], 0, grad[optim.split(" ")[0]][-1])
        trainer.perturb("theta")


    model.reset()
    epochs = [len(acc[optim.split(" ")[0]]) * 100,   len(acc[optim.split(" ")[1]]) * 100]
    seed = set_seed(seed)
    dset = DataSet(seed)
    optim = "{} {}".format(optim.split(" ")[0], optim.split(" ")[0])
    model = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes, optim=optim)
    trainer = Trainer(model, dset.x_train, dset.y_train, dset.x_test, dset.y_test, batch_size=dset.N_train)
    for i, opt in enumerate(optim.split(" ")):
        opt = opt + "_"
        train_fun = getattr(trainer.model, "train_{}".format(i + 1))
        acc[opt], grad[opt] = trainer.train(epochs=epochs[i], test_every=100, train_fun=train_fun)
        trainer.perturb("theta")

    # acc[opt], grad[opt] = trainer.train(epochs=epochs, test_every=100)

    save_dict(acc, value="acc flip")
    save_dict(grad, value="grad flip")

    visualizer.vanilla_plot(acc, x_every=100, value="Accuracy", flip=True)
    visualizer.vanilla_plot(grad, x_every=100, value="Squared Gradientsum", flip=True)
    visualizer.vanilla_plot(grad, x_every=100, value="Logarithm of Squared Gradientsum", flip=True, log=True)


def build_graphs(dset):
    model_r_simultnewton = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes,
                                       optim="SimultNewton")
    model_r_newton = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes,
                                    optim="Newton")
    model_r_spnewton = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes,
                                    optim="SPNewton")

    return model_r_simultnewton, model_r_spnewton, model_r_newton

def build_fast_graphs(dset):
    model_r_GD = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes,
                                       optim="GD")
    model_r_adam = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes,
                                    optim="Adam")

    return model_r_GD, model_r_adam


def single_train_run():
    dset = DataSet(np.random.randint(100))
    visualizer = Visualizer()
    acc = {}
    grad = {}
    seed = np.random.randint(0, 100)

    # # Vanilla Logistic Regression
    # tf.reset_default_graph()
    # set_seed(seed)
    # model = HSNet(input_size=dset.dim, out_size=dset.classes, batch_size=dset.N_train)
    # trainer = Trainer(model, dset.x_train, dset.y_train, dset.x_test, dset.y_test, batch_size=dset.N_train)
    # acc["Vanilla Opt"], grad["Vanilla Opt"] = trainer.train(5000, 100)

    # Robust Optimization Adam
    print("## Adam ##")
    tf.reset_default_graph()
    set_seed(seed)
    model = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes)
    trainer = Trainer(model, dset.x_train, dset.y_train, dset.x_test, dset.y_test, batch_size=dset.N_train)
    acc["Adam"], grad["Adam"] = trainer.train(2000, 100)

    # # Robust Optimization GD
    # print("## GD ##")
    # tf.reset_default_graph()
    # set_seed(seed)
    # model = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes, optim="GD")
    # trainer = Trainer(model, dset.x_train, dset.y_train, dset.x_test, dset.y_test, batch_size=dset.N_train)
    # acc["GD"], grad["GD"] = trainer.train(2000, 100)

    # Additive SPNewton Optimization
    # print("## Additive SPNewton ##")
    # tf.reset_default_graph()
    # set_seed(seed)
    # model_spp = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes, optim="ASPNewton")
    # trainer = Trainer(model_spp, dset.x_train, dset.y_train, dset.x_test, dset.y_test, batch_size=dset.N_train)
    # acc["ASPN"], grad["ASPN"] = trainer.train(3000, 100)

    # SPNewton Optimization
    print("## SPNewton ##")
    tf.reset_default_graph()
    set_seed(seed)
    model_spp = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim,
                                  out_size=dset.classes, optim="SPNewton")
    trainer = Trainer(model_spp, dset.x_train, dset.y_train, dset.x_test, dset.y_test, batch_size=dset.N_train)
    acc["SPN"], grad["SPN"] = trainer.train(1000, 100)

    # # Newton Optimization
    # print("## Newton ##")
    # tf.reset_default_graph()
    # set_seed(seed)
    # model_spp = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes, optim="Newton")
    # trainer = Trainer(model_spp, dset.x_train, dset.y_train, dset.x_test, dset.y_test, batch_size=dset.N_train)
    # acc["Newton"], grad["Newton"] = trainer.train(3000, 100)

    # Newton Optimization
    # print("## SimultNewton ##")
    # tf.reset_default_graph()
    # set_seed(seed)
    # model_spp = HSRobustNet(batch_size=dset.N_train, input_size=dset.dim, out_size=dset.classes, optim="SimultNewton")
    # trainer = Trainer(model_spp, dset.x_train, dset.y_train, dset.x_test, dset.y_test, batch_size=dset.N_train)
    # acc["SimultNewton"], grad["SimultNewton"] = trainer.train(1000, 100)

    visualizer.vanilla_plot(acc, x_every=100, value="Accuracy")
    visualizer.vanilla_plot(grad, x_every=100, value="Squared Gradientsum")


if __name__ == '__main__':
    main()