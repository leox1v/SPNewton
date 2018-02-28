import tensorflow as tf
import numpy as np

class SPNewtonOptimizer():
    def __init__(self, vars1, vars2, lr=0.01):
        self.vars = vars1 + vars2
        self.vars1 = vars1
        self.vars2 = vars2
        self.lr = tf.Variable(lr, dtype=tf.float32)
        self.num1 = self.get_num_weights(vars1)
        self.num2 = self.get_num_weights(vars2)


    def get_num_weights(self, params):
        shape = [layer.get_shape().as_list() for layer in params]
        n = 0
        for sh in shape:
            n_layer = 1
            for dim in sh:
                n_layer *= dim
            n += n_layer
        return n

    def step(self, loss1, loss2=None):
        ''' Loss1 is minimized over vars1, and Loss2 maximized over vars2'''
        if loss2 is None:
            loss2 = loss1

        H, J = self._compute_modified_Hessian(loss1, self.vars1, loss2, self.vars2)

        H_inv, e = self.inverse_Hessian(H)

        # Compute the Newton step
        grad = tf.expand_dims(J, 1)
        d = - tf.multiply(self.lr, tf.matmul(H_inv, grad))

        # Update parameters
        assign_ops = self._apply_step(d, self.vars)

        return assign_ops, e

    def _compute_modified_Hessian(self, loss1, vars1, loss2, vars2):
        H = []
        J1 = tf.gradients(loss1, vars1)
        J1 = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in J1], axis=0)
        J2 = tf.gradients(loss2, vars2)
        J2 = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in J2], axis=0)
        J = tf.concat([J1, J2], axis=0)
        num = J.get_shape().as_list()[0]
        vars = vars1 + vars2
        for i in range(num):
            second_deriv = tf.gradients(J[i], vars)
            second_deriv = [deriv_layer if deriv_layer is not None else tf.zeros(vars[j].get_shape().as_list()) for j, deriv_layer in enumerate(second_deriv)]
            second_deriv = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in second_deriv], axis=0)
            H.append(second_deriv)
        H = tf.convert_to_tensor(H, dtype=tf.float32)
        H = tf.multiply(tf.Variable(0.5, dtype=tf.float32), tf.add(H, tf.transpose(H)))
        return H, J

    def inverse_Hessian(self, H, thr=0.001):
        e, v = tf.self_adjoint_eig(H)

        #special eigval computation for the submatrices
        H_ul = tf.slice(H, [0,0], [self.num1, self.num1])
        e_ul, _ = tf.self_adjoint_eig(H_ul)
        H_lr = tf.slice(H, [self.num1, self.num1], [self.num2, self.num2])
        e_lr, _ = tf.self_adjoint_eig(H_lr)


        thr_condition = tf.greater(tf.abs(e), thr)
        sign_ = tf.cast(tf.greater(e, 0.0), tf.float32)
        e_inv = tf.where(thr_condition, tf.divide(1.0, e), tf.multiply(tf.multiply(tf.ones_like(e), 100.0), sign_))

        # SFN adjustment
        e_inv = tf.abs(e_inv)

        H_inv = tf.matmul(tf.matmul(v, tf.diag(e_inv)), tf.transpose(v))
        return H_inv, (e, e_ul, e_lr)

    def _apply_step(self, d, _vars):
        # Reshape d to fit the layer shapes
        shape = [layer.get_shape().as_list() for layer in _vars]
        d_list = []
        offset = 0
        for sh in shape:
            end = offset + np.prod(sh)
            _sign = 1
            # Change the sign of the direction for the parameters over which we maximize
            if offset >= self.num1:
                _sign = -1
            d_list.append(tf.reshape(_sign * d[offset:end], sh))
            offset = end

        # Do the assign operations
        assign_ops = []
        for i in range(len(_vars)):
            assign_ops.append(tf.assign_add(_vars[i], d_list[i]))

        return assign_ops

class AdditiveSPNewtoonOptimizer(SPNewtonOptimizer):

    def step(self, loss1, loss2=None):
        ''' Loss1 is minimized over vars1, and Loss2 maximized over vars2'''
        if loss2 is None:
            loss2 = loss1

        # compute full Hessian for both loss functions
        H1, J1 = self._compute_Hessian(loss1, self.vars)
        H2, J2 = self._compute_Hessian(loss2, self.vars)
        J = tf.concat([tf.slice(J1, [0], [self.num1]), tf.slice(J2, [self.num1], [self.num2])], axis=0)

        # SFN Hessian for both individual
        H1, e_ul = self._make_sfn(H1)
        H2, e_lr = self._make_sfn(H2)
        H = tf.multiply(tf.Variable(0.5, dtype=tf.float32), tf.add(H1, H2))

        H_inv, (e_full, _, _) = self.inverse_Hessian(H)
        e = (e_full, e_lr, e_ul)

        # Compute the Newton step
        grad = tf.expand_dims(J, 1)
        d = - tf.multiply(self.lr, tf.matmul(H_inv, grad))

        # Update parameters
        assign_ops = self._apply_step(d, self.vars)

        return assign_ops, e

    def _compute_Hessian(self, loss, _vars):
        H = []
        J = tf.gradients(loss, _vars)
        J = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in J], axis=0)
        num = J.get_shape().as_list()[0]
        for i in range(num):
            second_deriv = tf.gradients(J[i], _vars)
            second_deriv = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in second_deriv], axis=0)
            H.append(second_deriv)
        H = tf.convert_to_tensor(H, dtype=tf.float32)
        H = tf.multiply(tf.Variable(0.5, dtype=tf.float32), tf.add(H, tf.transpose(H)))
        return H, J

    def _make_sfn(self, H):
        e, v = tf.self_adjoint_eig(H)
        H_sfn = tf.matmul(tf.matmul(v, tf.diag(tf.abs(e))), tf.transpose(v))
        return H_sfn, e


class SimultSPNewtonOptimizer(SPNewtonOptimizer):

    def simult_step(self, loss1, loss2=None):
        ''' Loss1 is minimized over vars1, and Loss2 minimized over vars2'''
        if loss2 is None:
            loss2 = - loss1

        H1, J1 = self._compute_Hessian(loss1, self.vars1)
        H2, J2 = self._compute_Hessian(loss2, self.vars2)

        Hinv1, e_ul = self.inverse_Hessian(H1)
        Hinv2, e_lr = self.inverse_Hessian(H2)

        # Compute the Newton step
        grad1 = tf.expand_dims(J1, 1)
        d1 = - tf.multiply(self.lr, tf.matmul(Hinv1, grad1))
        grad2 = tf.expand_dims(J2, 1)
        d2 = - tf.multiply(self.lr, tf.matmul(Hinv2, grad2))

        # Update parameters
        assign_ops1 = self._apply_step(d1, self.vars1)
        assign_ops2 = self._apply_step(d2, self.vars2)

        return (assign_ops1, assign_ops2), (e_ul, e_ul, e_lr)

    def _compute_Hessian(self, loss, _vars):
        H = []
        J = tf.gradients(loss, _vars)
        J = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in J], axis=0)
        num = J.get_shape().as_list()[0]
        for i in range(num):
            second_deriv = tf.gradients(J[i], _vars)
            second_deriv = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in second_deriv], axis=0)
            H.append(second_deriv)
        H = tf.convert_to_tensor(H, dtype=tf.float32)
        H = tf.multiply(tf.Variable(0.5, dtype=tf.float32), tf.add(H, tf.transpose(H)))
        return H, J

    def inverse_Hessian(self, H, thr=0.001):
        e, v = tf.self_adjoint_eig(H)

        thr_condition = tf.greater(tf.abs(e), thr)
        sign_ = tf.cast(tf.greater(e, 0.0), tf.float32)
        e_inv = tf.where(thr_condition, tf.divide(1.0, e), tf.multiply(tf.multiply(tf.ones_like(e), 100.0), sign_))

        # SFN adjustment
        e_inv = tf.abs(e_inv)

        H_inv = tf.matmul(tf.matmul(v, tf.diag(e_inv)), tf.transpose(v))
        return H_inv, e

    def _apply_step(self, d, _vars):
        # Reshape d to fit the layer shapes
        shape = [layer.get_shape().as_list() for layer in _vars]
        d_list = []
        offset = 0
        for sh in shape:
            end = offset + np.prod(sh)
            d_list.append(tf.reshape(d[offset:end], sh))
            offset = end

        # Do the assign operations
        assign_ops = []
        for i in range(len(_vars)):
            assign_ops.append(tf.assign_add(_vars[i], d_list[i]))

        return assign_ops


class NewtonOptimizer:

    def __init__(self, vars1, vars2, lr=0.01):
        self.vars = vars1 + vars2
        self.num1 = self.get_num_weights(vars1)
        self.num2 = self.get_num_weights(vars2)
        self.lr = tf.Variable(lr, dtype=tf.float32)

    def step(self, loss):
        H, J = self._compute_Hessian(loss, self.vars)

        H_inv, e = self.inverse_Hessian(H)

        # Compute the Newton step
        grad = tf.expand_dims(J, 1)
        d = - tf.multiply(self.lr, tf.matmul(H_inv, grad))

        # Update parameters
        assign_ops = self._apply_step(d, self.vars)

        return assign_ops, e

    def _compute_Hessian(self, loss, _vars):
        H = []
        J = tf.gradients(loss, _vars)
        J = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in J], axis=0)
        num = J.get_shape().as_list()[0]
        for i in range(num):
            second_deriv = tf.gradients(J[i], _vars)
            second_deriv = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in second_deriv], axis=0)
            H.append(second_deriv)
        H = tf.convert_to_tensor(H, dtype=tf.float32)
        H = tf.multiply(tf.Variable(0.5, dtype=tf.float32), tf.add(H, tf.transpose(H)))
        return H, J

    def inverse_Hessian(self, H, thr=0.001):
        e, v = tf.self_adjoint_eig(H)

        # special eigval computation for the submatrices
        H_ul = tf.slice(H, [0, 0], [self.num1, self.num1])
        e_ul, _ = tf.self_adjoint_eig(H_ul)
        H_lr = tf.slice(H, [self.num1, self.num1], [self.num2, self.num2])
        e_lr, _ = tf.self_adjoint_eig(H_lr)

        thr_condition = tf.greater(tf.abs(e), thr)
        sign_ = tf.cast(tf.greater(e, 0.0), tf.float32)
        e_inv = tf.where(thr_condition, tf.divide(1.0, e), tf.multiply(tf.multiply(tf.ones_like(e), 100.0), sign_))

        H_inv = tf.matmul(tf.matmul(v, tf.diag(e_inv)), tf.transpose(v))

        return H_inv, (e, e_ul, e_lr)


    def _apply_step(self, d, _vars):
        # Reshape d to fit the layer shapes
        shape = [layer.get_shape().as_list() for layer in _vars]
        print(shape)
        d_list = []
        offset = 0
        for sh in shape:
            end = offset + np.prod(sh)
            d_list.append(tf.reshape(d[offset:end], sh))
            offset = end

        # Do the assign operations
        assign_ops = []
        for i in range(len(_vars)):
            assign_ops.append(tf.assign_add(_vars[i], d_list[i]))

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

class SimultNewtonOptimizer(NewtonOptimizer):

    def __init__(self, vars1, vars2, lr=0.01):
        super().__init__(vars1, vars2, lr)
        self.vars1 = vars1
        self.vars2 = vars2

    def simult_step(self, loss1, loss2=None):
        if loss2 == None:
            loss2 = loss1
        H1, J1 = self._compute_Hessian(loss1, self.vars1)
        H2, J2 = self._compute_Hessian(loss2, self.vars2)

        Hinv1, e_ul = self.inverse_Hessian(H1)
        Hinv2, e_lr = self.inverse_Hessian(H2)

        # Compute the Newton step
        grad1 = tf.expand_dims(J1, 1)
        d1 = - tf.multiply(self.lr, tf.matmul(Hinv1, grad1))

        grad2 = tf.expand_dims(J2, 1)
        d2 = - tf.multiply(self.lr, tf.matmul(Hinv2, grad2))

        # Update parameters
        assign_ops1 = self._apply_step(d1, self.vars1)
        assign_ops2 = self._apply_step(d2, self.vars2)

        return (assign_ops1, assign_ops2), (e_ul, e_ul, e_lr)


    def inverse_Hessian(self, H, thr=0.001):
        e, v = tf.self_adjoint_eig(H)

        thr_condition = tf.greater(tf.abs(e), thr)
        sign_ = tf.cast(tf.greater(e, 0.0), tf.float32)
        e_inv = tf.where(thr_condition, tf.divide(1.0, e), tf.multiply(tf.multiply(tf.ones_like(e), 100.0), sign_))

        H_inv = tf.matmul(tf.matmul(v, tf.diag(e_inv)), tf.transpose(v))

        return H_inv, e



# class NewtonOptimizer_():
#
#     def __init__(self, vars, lr=0.001, sfn=False, spp_max_idx_start=None, num_g = 1, num_d = 1, vars1=None, vars2=None):
#         self.vars = vars
#         self.lr = tf.Variable(lr, dtype=tf.float32)
#         self.sfn = sfn
#         self.spp_max_idx_start = spp_max_idx_start
#         self.num_g = num_g
#         self.num_d = num_d
#
#         self.vars1 = vars1
#         self.vars2 = vars2
#
#     def step(self, loss1, loss2):
#         # H, J = self._compute_Hessian(loss)
#         H, J = self._compute_modified_Hessian(loss1, self.vars1, loss2, self.vars2)
#
#         H_inv, rank, e = self.inverse_Hessian(H)
#
#         # Compute the Newton step
#         grad = tf.expand_dims(J, 1)
#         d = - tf.multiply(self.lr, tf.matmul(H_inv, grad))
#
#         # Update parameters
#         assign_ops = self._apply_step(d, self.vars)
#
#         return assign_ops, rank, e
#
#     def simul_step(self, loss1, loss2):
#         H, J = self._compute_modified_Hessian(loss1, self.vars1, loss2, self.vars2)
#         H_ul = tf.slice(H, [0, 0], [self.num_g, self.num_g])
#         H_lr = tf.slice(H, [self.num_g, self.num_g], [self.num_d, self.num_d])
#         e_ul, _ = tf.self_adjoint_eig(H_ul)
#         e_lr, _ = tf.self_adjoint_eig(H_lr)
#         J1 = tf.slice(J, [0], [self.num_g])
#         J2 = tf.slice(J, [self.num_g], [self.num_d])
#
#         H_inv1 = self.simple_inverse_Hessian(H_ul)
#         H_inv2 = self.simple_inverse_Hessian(H_lr)
#
#         # Compute the Newton step
#         grad1 = tf.expand_dims(J1, 1)
#         grad2 = tf.expand_dims(J2, 1)
#         d1 = - tf.multiply(self.lr, tf.matmul(H_inv1, grad1))
#         d2 = - tf.multiply(self.lr, tf.matmul(H_inv2, grad2))
#
#         if self.sfn:
#             d2 = - d2
#
#         # Update parameters
#         assign_ops1 = self._apply_step(d1, self.vars1)
#         assign_ops2 = self._apply_step(d2, self.vars2)
#
#         return (assign_ops1, assign_ops2), (e_ul, e_ul, e_lr)
#
#     def _compute_Hessian(self, loss):
#         H = []
#         J = tf.gradients(loss, self.vars)
#         J = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in J], axis=0)
#         num = J.get_shape().as_list()[0]
#         for i in range(num):
#             second_deriv = tf.gradients(J[i], self.vars)
#             second_deriv = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in second_deriv], axis=0)
#             H.append(second_deriv)
#         H = tf.convert_to_tensor(H, dtype=tf.float32)
#         H = tf.multiply(tf.Variable(0.5, dtype=tf.float32), tf.add(H, tf.transpose(H)))
#         return H, J
#
#     def _compute_modified_Hessian(self, loss1, vars1, loss2, vars2):
#         H = []
#         J1 = tf.gradients(loss1, vars1)
#         J1 = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in J1], axis=0)
#         J2 = tf.gradients(loss2, vars2)
#         J2 = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in J2], axis=0)
#         J = tf.concat([J1, J2], axis=0)
#         num = J.get_shape().as_list()[0]
#         vars = vars1 + vars2
#         for i in range(num):
#             second_deriv = tf.gradients(J[i], vars)
#             second_deriv = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in second_deriv], axis=0)
#             H.append(second_deriv)
#         H = tf.convert_to_tensor(H, dtype=tf.float32)
#         H = tf.multiply(tf.Variable(0.5, dtype=tf.float32), tf.add(H, tf.transpose(H)))
#         return H, J
#
#
#     def _apply_step(self, d, _vars):
#         # Reshape d to fit the layer shapes
#         shape = [layer.get_shape().as_list() for layer in _vars]
#         print(shape)
#         d_list = []
#         offset = 0
#         for sh in shape:
#             end = offset + np.prod(sh)
#             _sign = 1
#             if self.sfn and self.spp_max_idx_start is not None and self.spp_max_idx_start <= offset:
#                 _sign = -1
#             d_list.append(tf.reshape(_sign * d[offset:end], sh))
#             offset = end
#
#         # Do the assign operations
#         assign_ops = []
#         for i in range(len(_vars)):
#             assign_ops.append(tf.assign_add(_vars[i], d_list[i]))
#
#         return assign_ops
#
#     def _make_H_psd(self, H):
#         e, v = tf.self_adjoint_eig(H)
#         e_ = tf.abs(e)
#         H_ = tf.matmul(tf.matmul(v, tf.diag(e_)), tf.transpose(v))
#         return H_
#
#     def inverse_Hessian(self, H, thr=0.001):
#         e, v = tf.self_adjoint_eig(H)
#
#         #special eigval computation for the submatrices
#         H_ul = tf.slice(H, [0,0], [self.num_g, self.num_g])
#         e_ul, _ = tf.self_adjoint_eig(H_ul)
#         H_lr = tf.slice(H, [self.num_g, self.num_g], [self.num_d, self.num_d])
#         e_lr, _ = tf.self_adjoint_eig(H_lr)
#
#
#         thr_condition = tf.greater(tf.abs(e), thr)
#         sign_ = tf.cast(tf.greater(e, 0.0), tf.float32)
#         e_inv = tf.where(thr_condition, tf.divide(1.0, e), tf.multiply(tf.multiply(tf.ones_like(e), 100.0), sign_))#tf.zeros_like(e))
#         rank = tf.reduce_sum(tf.cast(thr_condition, tf.float32))
#
#         if self.sfn:
#             e_inv = tf.abs(e_inv)
#
#         H_inv = tf.matmul(tf.matmul(v, tf.diag(e_inv)), tf.transpose(v))
#         return H_inv, rank, (e, e_ul, e_lr)
#
#     def simple_inverse_Hessian(self, H, thr=0.001):
#         e, v = tf.self_adjoint_eig(H)
#         thr_condition = tf.greater(tf.abs(e), thr)
#         sign_ = tf.cast(tf.greater(e, 0.0), tf.float32)
#         e_inv = tf.where(thr_condition, tf.divide(1.0, e),
#                          tf.multiply(tf.multiply(tf.ones_like(e), 100.0), sign_))  # tf.zeros_like(e))
#
#         if self.sfn:
#             e_inv = tf.abs(e_inv)
#
#         H_inv = tf.matmul(tf.matmul(v, tf.diag(e_inv)), tf.transpose(v))
#         return H_inv




    # def debug_H(self, loss):
    #     H, _ = self._compute_Hessian_eff(loss)
    #     H_real, _ = self._compute_Hessian(loss)
    #     return H, H_real

    # def _compute_Hessian_eff(self, loss):
    #     J = tf.gradients(loss, self.vars)
    #     J = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in J], axis=0)
    #     n = J.get_shape().as_list()[0]
    #     H = [[] for _ in range(n)]
    #
    #     def get_var_index(i):
    #         ''' Gives you the index until which we need the second derivative'''
    #         shape = [layer.get_shape().as_list() for layer in self.vars]
    #         shape_cum = np.cumsum([np.prod(sh) for sh in shape])
    #         idx = next(idx for idx, v in enumerate(shape_cum) if v > i)
    #         throw_away = i
    #         if idx > 0:
    #             throw_away = i - shape_cum[idx - 1]
    #         return idx, throw_away
    #
    #     for i in range(n): # we iterate over all rows of the Hessian
    #         idx, throw_away = get_var_index(i)
    #         second_deriv = tf.gradients(J[i], self.vars[idx:])
    #         second_deriv = tf.concat([tf.reshape(deriv_layer, [-1]) for deriv_layer in second_deriv], axis=0)[throw_away:]
    #         H[i].append(second_deriv)
    #         for ii in range(i+1, n):
    #             H[ii].append([second_deriv[ii-i]])
    #     H = [tf.concat(row, axis=0) for row in H]
    #     H = tf.convert_to_tensor(H, dtype=tf.float32)
    #     return H, J
