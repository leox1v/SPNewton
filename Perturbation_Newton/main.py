import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from model import Model
import pprint
from utils import *
from utils import Helper
from dataset import DataSet
import pickle


flags = tf.app.flags
flags.DEFINE_string("max_iter", "200 200", "Maximum of iterations to train. [25]")
flags.DEFINE_string("batch_size", "8192", "The size of batch images [64]")

flags.DEFINE_string("dataset", "toy", "The dataset that is used. [toy, mnist, small_mnist]")
flags.DEFINE_integer("modes", 4, "The number of gaussian modes. [4]")

flags.DEFINE_integer("input_dim", 2, "The dimension of the input samples. [2]")
flags.DEFINE_integer("z_dim", 2, "The size of latent vector z.[256]")
flags.DEFINE_integer("D_h1", 30, "The hidden dimension of the first layer of the Discriminator. [10]")
flags.DEFINE_integer("G_h1", 30, "The hidden dimension of the first layer of the Generator. [10]")
flags.DEFINE_integer("D_h2", None, "The hidden dimension of the first layer of the Discriminator. [10]")
flags.DEFINE_integer("G_h2", None, "The hidden dimension of the first layer of the Generator. [10]")

flags.DEFINE_string("checkpoint_dir", "checkpoint/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("output_dir", "out/", "Directory name to save the image samples [samples]")
flags.DEFINE_string("summaries_dir", "tensorboard/", "Directory to use for the summary.")
flags.DEFINE_string("array_dir", "arr/", "Directory to use for arrays to store.")
flags.DEFINE_integer("imgs_to_print", 12, "The number of images that should be created. [12]")

flags.DEFINE_string("learning_rates", "0.05 0.05" , "Learning rates for the different opt_methods, respectively.")
flags.DEFINE_string("opt_methods", "newton spnewton", "Optimization methods that needs to be compared [adam extraGradient AdamSim altextraGradient]")

flags.DEFINE_boolean("compute_hessian", True, "Compute the full Hessian. Very Expensive!")
flags.DEFINE_string("regularizer", "None", "Regularization method that is used. [None, consensus]")

pp = pprint.PrettyPrinter()


def main(_):
    FLAGS = flags.FLAGS
    pp.pprint(flags.FLAGS.__flags)

    # load data
    data = DataSet(dataset=FLAGS.dataset, modes=FLAGS.modes, dim=FLAGS.input_dim)
    FLAGS = Helper(FLAGS).setup_directories()
    visualizer = Visualizer(FLAGS)

    visualizer.plot_gt(data)

    # Construct a model with the two optimizer
    seed = set_seed(40)
    model = Model(FLAGS, opt=FLAGS.opt_methods, learning_rate=FLAGS.learning_rates)
    trainer = Trainer(FLAGS, model, data, visualizer, compute_eigv=FLAGS.compute_hessian)

    trainer.initialize_session()
    trainer.train()
    trainer.reset()

    optimizer = "{} {}".format(FLAGS.opt_methods.split(" ")[0], FLAGS.opt_methods.split(" ")[0])
    learning_rates = "{} {}".format(FLAGS.learning_rates.split(" ")[0], FLAGS.learning_rates.split(" ")[0])
    set_seed(seed)
    model = Model(FLAGS, opt=optimizer, learning_rate=learning_rates)

    trainer.optimizer = [optimizer.split(" ")[0], optimizer.split(" ")[1] + "_"]
    trainer.model = model
    trainer.initialize_session()
    trainer.train(duplicate=True)


    save_dict(trainer.grad, FLAGS.array_dir, "grads")
    save_dict(trainer.critical_param_dist, FLAGS.array_dir, "critical_param_dist")
    save_dict(trainer.param_updates, FLAGS.array_dir, "param_updates")
    if FLAGS.compute_hessian:
        save_dict(trainer.eigenvalues, FLAGS.array_dir, "eigenvalues")

    trainer.plot()


def save_dict(_dict, array_dir, name):
    with open(os.path.join(array_dir, "{}.pickle".format(name)), 'wb') as handle:
        pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

class Trainer():
    def __init__(self, flags, model, data, visualizer, compute_eigv=False):
        self.iterations = [int(iter) for iter in flags.max_iter.split(" ")]
        self.batch_size = int(flags.batch_size)
        self.model = model
        self.data = data
        self.visualizer = visualizer
        self.session = None
        self.optimizer = flags.opt_methods.split(" ")
        self.grad = {}
        self.test_every = 100
        self.compute_eigv = compute_eigv
        self.eigenvalues = {'Full': {}, 'Generator': {}, 'Discriminator': {}}
        self.critical_param = None
        self.critical_param_dist = {}
        self.last_param = None
        self.param_updates = {}

    def initialize_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def reset(self):
        tf.reset_default_graph()

    def train(self, test_every=100, duplicate=False):
        self.test_every = test_every
        assert len(self.iterations) == 2, "Need exactly 2"

        step_fun = self.model.opt_step_1
        _optimizer = self.optimizer[0]

        for i in range(sum(self.iterations)):
            # Data batch
            X = self.data.next_batch(self.batch_size)
            Z = sample_Z(self.batch_size, self.model.z_dim)

            if i == self.iterations[0]:
                self._perturb_params()
                step_fun = self.model.opt_step_2
                _optimizer = self.optimizer[1]
                duplicate = False

            self.session.run(step_fun, feed_dict={self.model.X: X, self.model.Z: Z})

            if i % test_every == 0 and not duplicate:
                self._test(i)
                self._get_gradients(_optimizer)
                if self.compute_eigv:
                    self._compute_eigenvalues(i)
                if self.critical_param is not None:
                    self._distance_to_critical_param(i)
            if (i % 1 == 0 and not duplicate):
                self._get_update_dist(i)



    def _test(self, iteration):
        samples = 1000
        samples = self.session.run(self.model.g_z, feed_dict={self.model.Z: sample_Z(samples, self.model.z_dim)})

        #samples, data, samples_seen, iteration, optimizer
        optimizer = self.optimizer[0 if iteration < self.iterations[0] else 1]
        self.visualizer.plot_heat(samples, self.data, iteration=iteration, optimizer=optimizer)

    def _compute_eigenvalues(self, iteration):
        X = self.data.next_batch(self.batch_size)
        Z = sample_Z(self.batch_size, self.model.z_dim)

        (e, e_g, e_d) = self.session.run(self.model.eigvals, feed_dict={self.model.X: X, self.model.Z: Z})

        optimizer = self.optimizer[0 if iteration < self.iterations[0] else 1]
        self.eigenvalues["Full"] = safe_append(self.eigenvalues["Full"], optimizer, e)
        self.eigenvalues["Generator"] = safe_append(self.eigenvalues["Generator"], optimizer, e_g)
        self.eigenvalues["Discriminator"] = safe_append(self.eigenvalues["Discriminator"], optimizer, e_d)

    def _perturb_params(self):
        self.critical_param = self._get_model_param()
        self.session.run(self.model.perturb)

    def _get_model_param(self):
        theta_g, theta_d = self.session.run([self.model.theta_g, self.model.theta_d])
        param = theta_g + theta_d
        return param

    def _distance_to_critical_param(self, iteration):
        current_param = self._get_model_param()
        dist = np.sqrt(np.sum([np.sum(np.square(current_param[i] - layer)) for i,layer in enumerate(self.critical_param)]))
        optimizer = self.optimizer[0 if iteration < self.iterations[0] else 1]
        self.critical_param_dist = safe_append(self.critical_param_dist, optimizer, val=dist)
        return dist

    def _get_gradients(self, optimizer):
        N = min(self.data.N, 10000)
        X = self.data.next_batch(N)
        Z = sample_Z(N, self.model.z_dim)
        d_gradients, g_gradients = self.session.run([self.model.d_gradients, self.model.g_gradients],
                                                    feed_dict={self.model.X: X, self.model.Z: Z})
        self.grad = safe_append(self.grad, optimizer, val= d_gradients + g_gradients)

    def _get_update_dist(self, iteration):
        current_param = self._get_model_param()
        if self.last_param is not None:
            dist = np.sqrt(np.sum([np.sum(np.square(current_param[i] - layer)) for i, layer in enumerate(self.last_param)]))
            optimizer = self.optimizer[0 if iteration < self.iterations[0] else 1]
            self.param_updates = safe_append(self.param_updates, optimizer, val=dist)
        self.last_param = current_param

    def plot(self):
        self.visualizer.grad_plot(self.grad, self.test_every)
        self.visualizer.plot_eigenvalues(self.eigenvalues)
        self.visualizer.param_dist_plot(self.critical_param_dist)
        self.visualizer.param_update_plot(self.param_updates, test_every=1)


class Visualizer():
    def __init__(self, flags):
        self.flags = flags

    def plot_heat(self, samples, data, iteration="-", optimizer=""):
        fig, ax = plt.subplots()
        g = sns.kdeplot(samples[:, 0], samples[:, 1], shade=True, cmap='Greens', n_levels=20, ax=ax)
        ax.set_xlim([-1.5 * data.radius, 1.5 * data.radius])
        ax.set_ylim([-1.5 * data.radius, 1.5 * data.radius])
        ax.set_facecolor(sns.color_palette('Greens', n_colors=256)[0])
        ax.scatter([mean[0] for mean in data.means], [mean[1] for mean in data.means], c='r', marker="D")
        optimizer = optimizer.split("_")[0]

        ax.set_title("{}, Iteration: {}".format(optimizer.upper(), iteration))

        fig = g

        plt.savefig(self.flags.output_dir + optimizer + '_{}.png'.format(str(int(iteration)//100).zfill(3)))
        plt.close()
        return fig

    def grad_plot(self, _dict, test_every=100, log=True):
        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        offset = 0
        for i, key in enumerate(_dict.keys()):
            x_range = range(offset, offset + test_every*len(_dict[key]), test_every)
            if log:
                _dict[key] = save_log(_dict[key])
            plt.plot(x_range, _dict[key], label=key.replace("_", " cont'd"))

            if i == 0:
                offset += test_every*len(_dict[key])

        plt.axvline(x=offset, linewidth=1, color='k', label="Perturbation")

        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Logarithmic sum of squared gradients")

        plt.savefig(os.path.join(self.flags.output_dir, "../{}.png".format("Gradients")))
        plt.close()

    def plot_eigenvalues(self, eigenvalues):
        for part_key in eigenvalues.keys():
            eig = eigenvalues[part_key]
            # get the optimizer that is "duplicated"
            dup_opt = [key.split("_")[0] for key in eig.keys() if "_" in key][0]
            non_dup_opt = np.setdiff1d(list(eig.keys()), [dup_opt,dup_opt+"_"])[0]

            # construct the two arrays
            eig_same_opt = eig[dup_opt] + eig[dup_opt + "_"]
            eig_different_opt = eig[dup_opt] + eig[non_dup_opt]

            for ii, eig_array in enumerate([eig_same_opt, eig_different_opt]):
                n = len(eig_array)
                cols = np.arange(n)
                dim = len(eig_array[0])
                for i in cols:
                    plt.scatter([i*100] * dim, eig_array[i])
                opts = ["{} -> {}".format(dup_opt, dup_opt), "{} -> {}".format(dup_opt, non_dup_opt)][ii]
                plt.title("Eigenvalues of {} Hessian; {}".format(part_key, opts))
                plt.xlabel("Iterations")
                plt.savefig(os.path.join(self.flags.output_dir, "../eigenvalues_{}_{}.png".format(part_key, opts.replace(" -> ", ""))))
                plt.close()

    def param_dist_plot(self, _dict, test_every=100, log=True):
        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, key in enumerate(_dict.keys()):
            x_range = range(0, 0 + test_every*len(_dict[key]), test_every)
            if log:
                _dict[key] = save_log(_dict[key])
            plt.plot(x_range, _dict[key], label=key.replace("_", ""))

        plt.legend()
        plt.xlabel("Iterations after perturbation")
        plt.ylabel("Logarithmic euclidean distance to initial critical parameter")

        plt.savefig(os.path.join(self.flags.output_dir, "../{}.png".format("ParamDistance")))
        plt.close()

    def param_update_plot(self, _dict, test_every=100, log=True):
        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        offset = test_every
        for i, key in enumerate(_dict.keys()):
            x_range = range(offset, offset + test_every*len(_dict[key]), test_every)
            if log:
                _dict[key] = save_log(_dict[key])
            plt.plot(x_range, _dict[key], label=key.replace("_", " cont'd"))

            if i == 0:
                offset += test_every*(len(_dict[key]))

        plt.axvline(x=offset-test_every, linewidth=1, color='k', label="Perturbation")

        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Logarithm of norm of update")

        plt.savefig(os.path.join(self.flags.output_dir, "../{}.png".format("Updates")))
        plt.close()

    def plot_gt(self, data):
        if self.flags.dataset == "toy":
            samples = data.next_batch(1000)

            fig, ax = plt.subplots()
            g = sns.kdeplot(samples[:, 0], samples[:, 1], shade=True, cmap='Greens', n_levels=20, ax=ax)
            ax.set_xlim([-1.5 * data.radius, 1.5 * data.radius])
            ax.set_ylim([-1.5 * data.radius, 1.5 * data.radius])
            ax.set_facecolor(sns.color_palette('Greens', n_colors=256)[0])
            ax.scatter([mean[0] for mean in data.means], [mean[1] for mean in data.means], c='r', marker="D")

            plt.savefig(self.flags.output_dir + '_gt.png')
            plt.close()


if __name__ == '__main__':
    tf.app.run()