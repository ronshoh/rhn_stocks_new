from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell


class Model(object):
    """A Variational RHN model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.depth = depth = config.depth
        self.size = size = config.hidden_size
        self.num_layers = num_layers = config.num_layers
        self.num_of_features = num_of_features = config.num_of_features

        self.in_size = rhn_in_size = config.num_of_features

        self.out_size = out_size = config.out_size

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps, num_of_features])
        self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])
        self._mask = tf.placeholder(tf.float32, [batch_size, num_steps])
        # self._noise_x = tf.placeholder(tf.float32, [batch_size, num_steps, 1])
        self._noise_i = tf.placeholder(tf.float32, [batch_size, rhn_in_size, num_layers])
        self._noise_h = tf.placeholder(tf.float32, [batch_size, size, num_layers])
        self._noise_o = tf.placeholder(tf.float32, [batch_size, 1, size])

        inputs = self._input_data

        # W_in_mat = tf.get_variable("W_in_mat", [num_of_features, size])


        outputs = []
        self._initial_state = [0] * self.num_layers
        state = [0] * self.num_layers
        self._final_state = [0] * self.num_layers
        for l in range(config.num_layers):
            with tf.variable_scope('RHN' + str(l)):
                cell = RHNCell(size, rhn_in_size, is_training, depth=depth, forget_bias=config.init_bias)
                self._initial_state[l] = cell.zero_state(batch_size, tf.float32)
                state[l] = [self._initial_state[l], self._noise_i[:, :, l], self._noise_h[:, :, l]]
                for time_step in range(num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state[l]) = cell(inputs[:, time_step, :], state[l], config.state_gate)
                    outputs.append(cell_output)
                inputs = tf.stack(outputs, axis=1)
                outputs = []

        output = tf.reshape(inputs * self._noise_o, [-1, size])

        w_out_mat =  tf.get_variable("w_out_mat", [size, out_size])
        b_out_mat = tf.get_variable("b_out_mat", [out_size], initializer=tf.zeros_initializer())

        scores = tf.matmul(output, w_out_mat) + b_out_mat

        self._predictions = tf.reshape(scores, [batch_size, num_steps])

        weights = self._mask
        # if config.std_normalization:
        #     rel_tars = self._targets[tf.equal(weights,1.0)]
        #     _, var_tars = tf.nn.moments(rel_tars, axes=[0])
        #     rel_pred = self._predictions[tf.equal(weights,1.0)]
        #     _, var_pred = tf.nn.moments(rel_pred, axes=[0])
        #     weights = weights/(tf.sqrt(var_tars*var_pred))

        loss = tf.losses.mean_squared_error(
            self._targets,
            self._predictions,
            weights=weights,
            scope="MSE"
        )


        pred_loss = tf.reduce_sum(loss) # / (tf.reduce_sum(self._mask) + epsilon)

        self._cost = cost = pred_loss

        self._final_state = [s[0] for s in state]

        if not is_training:
            self._global_norm = tf.constant(0.0, dtype=tf.float32)
            self._l2_loss = tf.constant(0.0, dtype=tf.float32)
            return

        self._tvars = tf.trainable_variables()
        self._l2_loss = l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self._tvars])
        self._cost = cost = pred_loss + config.weight_decay * l2_loss

        self._lr = tf.Variable(0.0, trainable=False)
        self._nvars = np.prod(self._tvars[0].get_shape().as_list())
        print(self._tvars[0].name, self._tvars[0].get_shape().as_list())
        for var in self._tvars[1:]:
            sh = var.get_shape().as_list()
            print(var.name, sh)
            self._nvars += np.prod(sh)
        print(self._nvars, 'total variables')
        grads, self._global_norm = tf.clip_by_global_norm(tf.gradients(cost, self._tvars),
                                          config.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, self._tvars))

        with tf.variable_scope('optimizers'):
            if config.adaptive_optimizer == "Adam":
                optimizer_ad = tf.train.AdamOptimizer(self.lr)
            elif config.adaptive_optimizer == "RMSProp":
                optimizer_ad = tf.train.RMSPropOptimizer(self.lr)
            else:
                print("invalid optimizer option.. exiting!")
                optimizer_ad = []
                exit()
            self._train_op_ad = optimizer_ad.apply_gradients(zip(grads, self._tvars))

            with tf.variable_scope('ASGD'):
                self._counter = tf.Variable(0.0, trainable=False)
                optimizer_sgd = tf.train.GradientDescentOptimizer(self.lr)

                self._final_weights = []
                self._temp_weights = []
                for var in self._tvars:
                    self._final_weights.append(tf.get_variable(var.op.name + '_final',
                                                               initializer=tf.zeros_like(var, dtype=tf.float32),
                                                               trainable=False))
                    self._temp_weights.append(tf.get_variable(var.op.name + '_temp',
                                                              initializer=tf.zeros_like(var, dtype=tf.float32),
                                                              trainable=False))



                self._train_op_sgd = optimizer_sgd.apply_gradients(zip(grads, self._tvars))

                adder = tf.Variable(1.0, trainable=False)
                self._add_counter_op = tf.assign_add(self._counter, adder)
                self._asgd_acc_op = [tf.assign_add(self._final_weights[i], var) for i, var in
                                     enumerate(self._tvars)]
                self._reset_accumulation_op = [tf.assign(self._final_weights[i], tf.zeros_like(var)) for i, var in
                                               enumerate(self._final_weights)]

                self._set_asgd_weights = [tf.assign(self._tvars[i], tf.divide(var, self._counter)) for i, var
                                          in enumerate(self._final_weights)]
                self._store_weights = [tf.assign(self._temp_weights[i], var) for i, var in enumerate(self._tvars)]

                self._return_regular_weights = [tf.assign(self._tvars[i], var) for i, var
                                                in enumerate(self._temp_weights)]



    def reset_asgd(self, session):
        counter = session.run(self.counter)
        session.run(tf.assign(self.counter, counter * 0))
        session.run(self.reset_accumulation_op)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def store_set_asgd_weights(self, session):
        session.run(self.store_weights)
        session.run(self.set_asgd_weights)

    @property
    def add_counter_op(self):
        return self._add_counter_op

    @property
    def asgd_acc_op(self):
        return self._asgd_acc_op

    @property
    def return_regular_weights(self):
        return self._return_regular_weights

    @property
    def reset_accumulation_op(self):
        return self._reset_accumulation_op

    @property
    def store_weights(self):
        return self._store_weights

    @property
    def set_asgd_weights(self):
        return self._set_asgd_weights

    @property
    def counter(self):
        return self._counter

    @property
    def final_weights(self):
        return self._final_weights

    @property
    def temp_weights(self):
        return self._temp_weights

    @property
    def tvars(self):
        return self._tvars

    @property
    def predictions(self):
        return self._predictions

    @property
    def l2_loss(self):
        return self._l2_loss

    @property
    def global_norm(self):
        return self._global_norm

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def mask(self):
        return self._mask

    @property
    def noise_i(self):
        return self._noise_i

    @property
    def noise_h(self):
        return self._noise_h

    @property
    def noise_o(self):
        return self._noise_o

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op_ad(self):
        return self._train_op_ad

    @property
    def train_op_sgd(self):
        return self._train_op_sgd


    @property
    def nvars(self):
        return self._nvars


class RHNCell(RNNCell):
    """Variational Recurrent Highway Layer

  Reference: https://arxiv.org/abs/1607.03474
  """

    def __init__(self, num_units, in_size, is_training, depth=3, forget_bias=None):
        self._num_units = num_units
        self._in_size = in_size
        self.is_training = is_training
        self.depth = depth
        self.forget_bias = forget_bias

    @property
    def input_size(self):
        return self._in_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, st_gate=False, scope=None):
        former_state = current_state = state[0]
        noise_i = state[1]
        noise_h = state[2]
        for i in range(self.depth):
            with tf.variable_scope('h_' + str(i)):
                if i == 0:
                    h = tf.tanh(linear([inputs * noise_i, current_state * noise_h], self._num_units, True))
                else:
                    h = tf.tanh(linear([current_state * noise_h], self._num_units, True))
            with tf.variable_scope('t_' + str(i)):
                if i == 0:
                    t = tf.sigmoid(
                        linear([inputs * noise_i, current_state * noise_h], self._num_units, True, self.forget_bias))
                else:
                    t = tf.sigmoid(linear([current_state * noise_h], self._num_units, True, self.forget_bias))
            current_state = (h - current_state) * t + current_state

        if st_gate:
            print('# using state gating #')
            with tf.variable_scope('state_gate'):
                g = tf.sigmoid(linear(
                    [former_state * noise_h, current_state * noise_h], self._num_units, True, self.forget_bias * 1))
                current_state = g * former_state + (1 - g) * current_state

        return current_state, [current_state, noise_i, noise_h]


def linear(args, output_size, bias, bias_start=None, scope=None):
    """
  This is a slightly modified version of _linear used by Tensorflow rnn.
  The only change is that we have allowed bias_start=None.

  Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), matrix)
        if not bias:
            return res
        elif bias_start is None:
            bias_term = vs.get_variable("Bias", [output_size], dtype=dtype)
        else:
            bias_term = vs.get_variable("Bias", [output_size], dtype=dtype,
                                        initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return res + bias_term
