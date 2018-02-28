import numpy as np
from scipy.stats import multivariate_normal
from tensorflow.examples.tutorials.mnist import input_data
from skimage.measure import block_reduce
from sklearn.datasets import load_digits

class DataSet():
    def __init__(self, dataset,radius=2.0, std=0.005, modes=8, dim=2):
        self.dataset = dataset
        if self.dataset == "toy":
            self.modes = modes
            self.dim = dim
            self.radius = radius
            self.std = std
            self.means = self.generate_fake_distribution()
            self.sample_data = self.next_batch(10000)
            self.N = 100000
        elif self.dataset == "mnist":
            self.data = input_data.read_data_sets("MNIST_data/", one_hot=True)
            self.modes = 10
            self.dim = dim
            self.N = 60000
            if self.dim == 784:
                self.block_red_size = (1,1,1)
                self.height = 28
                self.width = 28
            elif self.dim == 196:
                self.block_red_size = (1,2,2)
                self.height = 14
                self.width = 14
        elif self.dataset == "small_mnist":
            self.data = load_digits()
            self.height = 8
            self.width = 8
            self.N = len(self.data.target)

        else:
            raise NotImplementedError

    def generate_fake_distribution(self):
        thetas = np.linspace(0, 2 * np.pi, self.modes+1)
        xs, ys = self.radius * np.sin(thetas[:self.modes]), self.radius * np.cos(thetas[:self.modes])
        loc = np.vstack([xs, ys]).T

        # Grid
        # x, y = np.linspace(-self.radius, self.radius, int(np.sqrt(self.modes))), np.linspace(-self.radius, self.radius, int(np.sqrt(self.modes)))
        # X, Y = np.meshgrid(x, y)
        # loc = np.vstack(list(zip(np.ravel(X), np.ravel(Y))))


        return loc

    def get_pdf(self, X):
        """
        Function to query the probability density function of the mixture of gaussians.
        :param X: with shape (n_samples, 2). Points for which we want to query the pdf function of the mixture.
        :return: Array of size n_samples with the function values for X.
        """
        sums = np.zeros(np.array(X).shape[0])
        for n, x in enumerate(X):
            for i in range(self.modes):
                sums[n] += 1.0 / self.modes * multivariate_normal.pdf(x, self.means[i], self.std)
        return sums


    def next_batch(self, batch_size, test=False):
        if self.dataset == "toy":
            rand_modes = np.random.choice(self.modes, batch_size)
            mean = np.array([self.means[mode] for mode in rand_modes])
            samples = np.zeros([batch_size, self.dim])
            for i in range(batch_size):
                samples[i, :] = np.random.multivariate_normal(mean[i], [[self.std,0], [0,self.std]])
            return samples
        elif self.dataset == "mnist":
            if test:
                samples, labels = self.data.test.next_batch(batch_size)
                return self.downsample(samples), labels
            else:
                samples, labels = self.data.train.next_batch(batch_size)
                return self.downsample(samples)
        elif self.dataset == "small_mnist":
            idx = np.random.choice(len(self.data.target), size=batch_size, replace=False)
            return self.data.data[idx, :]


    def downsample(self, samples):
        if self.block_red_size == (1,1,1):
            return samples
        samples = np.reshape(samples, [-1, 28, 28])
        samples = block_reduce(samples, block_size=self.block_red_size, func=np.mean)
        samples = np.reshape(samples, [-1, samples.shape[1] * samples.shape[2]])
        return samples



    def sorted_batch(self, size_per_label):
        if self.dataset == "toy":
            labels = np.repeat(range(self.modes), size_per_label)
            mean = np.array([self.means[mode] for mode in labels])
            samples = np.zeros([len(labels), self.dim])
            for i in range(len(labels)):
                samples[i, :] = np.random.multivariate_normal(mean[i], [[self.std,0], [0,self.std]])
            return samples, labels
        elif self.dataset == "mnist":
            completed = False
            sorted_batch = np.zeros([self.modes, size_per_label, self.dim])
            counter = np.zeros(self.modes).astype(int)

            while not completed:
                imgs, labels = self.next_batch(100, test=True)
                labels = np.argmax(labels, axis=1).astype(int)
                for i, label in enumerate(labels):
                    if counter[label] < size_per_label:
                        sorted_batch[label, counter[label], :] = imgs[i]
                        counter[label] += 1
                # print("Not yet completed! with: {}".format(counter))
                completed = np.all(counter >= size_per_label)
            sorted_batch = np.reshape(sorted_batch, [-1, self.dim])
            labels = np.repeat(range(self.modes), size_per_label)
            return sorted_batch, labels



