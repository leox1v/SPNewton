import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from newton import NewtonOptimizer, SPNewtonOptimizer, SimultNewtonOptimizer, AdditiveSPNewtoonOptimizer


class Model():
    def __init__(self, flags, opt="adam sgd", learning_rate="0.001 0.001"):
        self.opt = opt
        self.FLAGS = flags
        self.img_dim = self.FLAGS.input_dim
        self.z_dim = self.FLAGS.z_dim
        self.out_img_dim = self.FLAGS.input_dim

        # Placeholder
        self.X = tf.placeholder(tf.float32, shape=[None, self.img_dim], name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.FLAGS.z_dim], name='Z')

        self.g_z, self.theta_g, self.theta_d, self.D_loss, self.G_loss, self.loss_N = self.training_procedure()

        assert len(opt.split(" ")) == 2, "Need to give two optimizer"

        self.opt_step_1, self.eigvals = self.choose_optimizer(opt.split(" ")[0], float(learning_rate.split(" ")[0]), flags.compute_hessian)
        self.opt_step_2, _ = self.choose_optimizer(opt.split(" ")[1], float(learning_rate.split(" ")[1]), flags.compute_hessian)

        self.g_gradients = self.gradient_norm(-self.D_loss, self.theta_g)
        self.d_gradients = self.gradient_norm(-self.D_loss, self.theta_d)

        self.perturb = self.perturb_params(self.theta_g + self.theta_d)


    def choose_optimizer(self, opt, learning_rate, compute_hessian=False):
        eigvals = None
        if "newton" in opt:
            if opt == "spnewton":
                newton_opt = SPNewtonOptimizer(self.theta_g, self.theta_d, learning_rate)
                opt_step, eigvals = newton_opt.step(-self.D_loss)
                # self.newton_step, self.eigvals = newton_opt.step(self.loss_N)

            elif opt == "newton":
                newton_opt = NewtonOptimizer(self.theta_g, self.theta_d, learning_rate)
                opt_step, eigvals = newton_opt.step(-self.D_loss)

            elif opt == "simultnewton":
                newton_opt = SimultNewtonOptimizer(self.theta_g, self.theta_d, learning_rate)
                opt_step, eigvals = newton_opt.simult_step(-self.D_loss)

            elif opt == "additivespnewton":
                newton_opt = AdditiveSPNewtoonOptimizer(self.theta_g, self.theta_d, learning_rate)
                opt_step, eigvals = newton_opt.step(self.G_loss, -self.D_loss)
            else:
                raise NotImplementedError

        else:
            if opt == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif opt == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            else:
                raise NotImplementedError

            opt_step_g, g_gradients, g_grads_vars = self.minimize(optimizer, self.G_loss, self.theta_g)
            opt_step_d, d_gradients, d_grads_vars = self.minimize(optimizer, self.D_loss, self.theta_d)
            opt_step = (opt_step_g, opt_step_d)

            if compute_hessian:
                eigvals = self.compute_eigvals(self.G_loss,  self.theta_g, -self.D_loss, self.theta_d)

        return opt_step, eigvals

    def perturb_params(self, param_list):
        assign_ops = []
        for param in param_list:
            perturb_tensor = tf.random_normal(param.shape, mean=0.0, stddev=.1)
            assign_ops.append(tf.assign_add(param, perturb_tensor))
        return assign_ops

    def get_num_weights(self, params):
        shape = [layer.get_shape().as_list() for layer in params]
        n = 0
        for sh in shape:
            n_layer = 1
            for dim in sh:
                n_layer *= dim
            n += n_layer
        return n

    def gradient_norm(self, loss, vars):
        gradients = tf.gradients(loss, vars)
        gradients = tf.concat([tf.reshape(grad, [-1]) for grad in gradients], 0)
        grad_norm = tf.reduce_sum(tf.square(gradients))
        return grad_norm

    def minimize(self, optimizer, loss, var_list):
        grads_vars = optimizer.compute_gradients(loss, var_list=var_list)
        list_of_gradmat = tf.concat([tf.reshape(gv[0], [-1]) for gv in grads_vars], 0)
        squared_gradients = tf.reduce_sum(tf.square(list_of_gradmat))
        solver = optimizer.apply_gradients(grads_vars)
        return solver, squared_gradients, grads_vars

    def training_procedure(self):
        # Training Procedure
        g_z, theta_g = self.generator(self.Z)
        d_logit_real, theta_d = self.discriminator(self.X)
        d_logit_synth, _ = self.discriminator(g_z, reuse=True)

        # Loss for Discriminator: - log D(x) - log(1-D(G(z)))
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_synth, labels=tf.zeros_like(d_logit_synth)))

        D_loss = D_loss_real + D_loss_fake

        # Loss for Generator: -log(D(G(z)))
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_synth, labels=tf.ones_like(d_logit_synth)))

        # Numerics of GANs loss: log D(x) - log(D(G(z)))
        loss_N = -D_loss_real + G_loss

        return g_z, theta_g, theta_d, D_loss, G_loss, loss_N

    def generator(self, z, reuse=False):
        scope = "generator"
        with tf.variable_scope(scope):
            g_hidden = tf.layers.dense(z, self.FLAGS.G_h1, activation=tf.nn.relu, kernel_initializer=xavier_initializer(), name="G1", reuse=reuse)
            if self.FLAGS.G_h2 is not None:
                g_hidden = tf.layers.dense(g_hidden, self.FLAGS.G_h2, activation=tf.nn.relu, kernel_initializer=xavier_initializer(), name="Gh2", reuse=reuse)
            g_z = tf.layers.dense(g_hidden, self.img_dim, kernel_initializer=xavier_initializer(), name="G2", reuse=reuse)
            if self.FLAGS.dataset == "mnist":
                g_z = tf.nn.sigmoid(g_z, name="g_z")
            else:
                g_z = tf.identity(g_z, name="g_z")
        theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return g_z, theta_g


    def discriminator(self, x, reuse=False):
        scope = "discriminator"
        with tf.variable_scope(scope):
            d_hidden = tf.layers.dense(x, self.FLAGS.D_h1, activation=tf.nn.relu, kernel_initializer=xavier_initializer(), reuse=reuse, name="D1")
            if self.FLAGS.D_h2 is not None:
                d_hidden = tf.layers.dense(d_hidden, self.FLAGS.D_h2, activation=tf.nn.relu,
                                           kernel_initializer=xavier_initializer(), name="Dh2", reuse=reuse)
            d_logit = tf.layers.dense(d_hidden, 1, kernel_initializer=xavier_initializer(), reuse=reuse, name="D2")
        theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return d_logit, theta_d


    def reset_graph(self):
        tf.reset_default_graph()


    def _compute_Hessian(self, loss, vars):
        H = []
        J = tf.gradients(loss, vars)
        J = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in J], axis=0)
        num = J.get_shape().as_list()[0]
        for i in range(num):
            second_deriv = tf.gradients(J[i], vars)
            second_deriv = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in second_deriv], axis=0)
            H.append(second_deriv)
        H = tf.convert_to_tensor(H, dtype=tf.float32)
        # H = tf.multiply(tf.Variable(0.5, dtype=tf.float32), tf.add(H, tf.transpose(H)))
        return H, J



    def compute_eigvals(self, loss1, vars1, loss2, vars2):
        H1,J1 = self._compute_Hessian(loss1, vars1)
        H2, J2 = self._compute_Hessian(loss2, vars2)

        e1, _ = tf.self_adjoint_eig(H1)
        e2, _ = tf.self_adjoint_eig(H2)

        return (e1, e1, e2)