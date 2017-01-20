import gym

env = gym.make('Pendulum-v0')
env.reset(), env.render()  # HACK - bug with rendering if called after tensorflow import :( just started

from tqdm import tqdm
from collections import deque
import numpy as np
from scipy import stats

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops

import matplotlib.pyplot as plt
from noise import pnoise1
import cv2
#cv2.startWindowThread()
#cv2.namedWindow("criticactor", cv2.WINDOW_NORMAL)

FUTURE_THETA_N = 1
TIME_DIM = 1

TAU = 0.01
GAMMA = 0.99
ACTOR_LR = 0.001
CRITIC_LR = 0.001
ACTOR_L2_WEIGHT_DECAY = 0.00
CRITIC_L2_WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100000
MAX_EPISODE_LENGTH = 300
ITERATIONS_BEFORE_TRAINING = BATCH_SIZE + 1 + TIME_DIM
BETA = 0.0

TRAJECTORY_MODEL_NUM_MODELS = 1
TRAJECTORY_MODEL_NUM_PARAMETERS = 4

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
THETA_PHI_PER_ACTION_DIM = TRAJECTORY_MODEL_NUM_MODELS * TRAJECTORY_MODEL_NUM_PARAMETERS
FLATTENED_THETA_PHI_DIM = THETA_PHI_PER_ACTION_DIM * ACTION_DIM


def phi(theta):
    split_parameters = tf.split(2, TRAJECTORY_MODEL_NUM_PARAMETERS, tf.reshape(theta, [-1, ACTION_DIM, TRAJECTORY_MODEL_NUM_PARAMETERS, THETA_PHI_PER_ACTION_DIM // TRAJECTORY_MODEL_NUM_PARAMETERS]))
    frequency, phase, amplitude, time = split_parameters
    phase = 0
    return tf.reduce_sum(amplitude * tf.sign(tf.cos(np.pi * frequency**5.0 * time + 2*np.pi*phase)), axis=[1, 2, 3])


def phishift(theta):
    split_parameters = tf.split(2, TRAJECTORY_MODEL_NUM_PARAMETERS, tf.reshape(theta, [-1, ACTION_DIM, TRAJECTORY_MODEL_NUM_PARAMETERS, THETA_PHI_PER_ACTION_DIM // TRAJECTORY_MODEL_NUM_PARAMETERS]))
    split_parameters[3] = tf.mod(split_parameters[3] - 1, 2 * np.pi / split_parameters[0])  # time
    return tf.reshape(tf.concat(2, split_parameters), [-1, ACTION_DIM * THETA_PHI_PER_ACTION_DIM])


def fanin_init(layer):
    fanin = layer.get_shape().as_list()[1]
    v = 1.0 / np.sqrt(fanin)
    return tf.random_uniform_initializer(minval=-v, maxval=v)


def bounded_constraint(forward_op, min=-1.0, max=1.0):
    min, max = float(min), float(max)
    gradient_op_name = "BoundedConstraint-" + str(np.random.randint(1000, 9999))

    @tf.RegisterGradient(gradient_op_name)
    def plus_minus_one_gradient(op, grad):
        # Inverting Gradients for Bounded Control - https://www.cs.utexas.edu/~AustinVilla/papers/ICLR16-hausknecht.pdf
        return [grad * tf.select(grad < 0.0, (max - op.outputs[0]) / (max-min), (op.outputs[0] - min) / (max-min))]

    with forward_op.graph.gradient_override_map({"Identity": gradient_op_name}):
        return tf.identity(forward_op)


def passthrough_multiply(forward_op_a, forward_op_b):
    # Useful note: https://github.com/vanandrew/softmax_regression_segmentation/blob/06b548fffe4055bf843ec708cf6543c6b6fc3e49/lib/python2.7/site-packages/tensorflow/python/ops/math_grad.py
    gradient_op_name = "AgnosticMultiplier-" + str(np.random.randint(1000, 9999))

    @tf.RegisterGradient(gradient_op_name)
    def backward_agnostic_gradient(op, grad):
        x = op.inputs[0]
        y = op.inputs[1]
        assert x.dtype.base_dtype == y.dtype.base_dtype, (x.dtype, " vs. ", y.dtype)
        sx = array_ops.shape(x)
        sy = array_ops.shape(y)
        rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
        return (array_ops.reshape(math_ops.reduce_sum(grad, rx), sx),
                array_ops.reshape(math_ops.reduce_sum(grad, ry), sy))

    with forward_op_a.graph.gradient_override_map({"Mul": gradient_op_name}):
        return tf.mul(forward_op_a, forward_op_b)

class Noise:
    def __init__(self, ):
        global FLATTENED_THETA_PHI_DIM
        self.state = np.zeros(FLATTENED_THETA_PHI_DIM)
        self.perlin_seed = np.random.randint(1, 20, size=FLATTENED_THETA_PHI_DIM)

    def josh(self, mu, sigma):
        return stats.truncnorm.rvs((-1 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma, size=len(self.state))

    def perlin(self, step):
        return np.array([pnoise1(step * self.perlin_seed[i] / 256.0 - 0.5 * self.perlin_seed[i], 3) for i in range(FLATTENED_THETA_PHI_DIM)])

    def ou(self, theta, sigma):
        self.state += theta * self.state - sigma * np.random.randn(len(self.state))
        return self.state

class ActorCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, outer_scope, reuse=False):
        self.outer_scope = outer_scope
        self.reuse = reuse
        self._state_is_tuple = True

    @property
    def state_size(self):
        return (FLATTENED_THETA_PHI_DIM, FLATTENED_THETA_PHI_DIM)

    @property
    def output_size(self):
        return FLATTENED_THETA_PHI_DIM

    def __call__(self, inputs, state, scope=None):
        last_theta_phi = state[0]
        with tf.variable_scope(self.outer_scope + '/actor', reuse=self.reuse):
            uniform_random = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
            inputs = tf.concat(1, [inputs, last_theta_phi])
            net = slim.fully_connected(inputs, 500, weights_initializer=fanin_init(inputs),
                                       biases_initializer=fanin_init(inputs))  # activation_fn=lambda x: tf.maximum(0.2*x, x)
            residual = slim.fully_connected(net, FLATTENED_THETA_PHI_DIM, weights_initializer=uniform_random,
                                            biases_initializer=uniform_random, activation_fn=None)

            spark = bounded_constraint(slim.fully_connected(inputs, FLATTENED_THETA_PHI_DIM, weights_initializer=uniform_random, biases_initializer=tf.constant_initializer(0.01), activation_fn=None), min=0, max=1)

            with spark.graph.gradient_override_map({"Sign": "Identity"}):
                residual_filtered = passthrough_multiply(residual, 1.0)  # (tf.sign(tf.abs(residual) - spark) + 1) / 2.0

            # Add bounds
            output = []
            split_phi = tf.split(2, TRAJECTORY_MODEL_NUM_PARAMETERS, tf.reshape(last_theta_phi, [-1, ACTION_DIM, TRAJECTORY_MODEL_NUM_PARAMETERS, THETA_PHI_PER_ACTION_DIM // TRAJECTORY_MODEL_NUM_PARAMETERS]))
            split_residual = tf.split(2, TRAJECTORY_MODEL_NUM_PARAMETERS, tf.reshape(residual_filtered, [-1, ACTION_DIM, TRAJECTORY_MODEL_NUM_PARAMETERS, THETA_PHI_PER_ACTION_DIM // TRAJECTORY_MODEL_NUM_PARAMETERS]))

            output.append(bounded_constraint(split_phi[0] + split_residual[0], min=0, max=.2))   # frequency
            output.append(split_phi[1] + bounded_constraint(split_residual[1], min=-1, max=1))  # phase
            output.append(bounded_constraint(split_phi[2] + split_residual[2], min=0, max=1))   # amplitude
            output.append(split_phi[3])                                                         # time

            output = tf.reshape(tf.concat(2, output), [-1, ACTION_DIM * THETA_PHI_PER_ACTION_DIM])

        return output, (spark, residual)


def critic_network(state, theta_phi, last_theta_phi, outer_scope, reuse=False):
    with tf.variable_scope(outer_scope + '/critic', reuse=reuse):
        uniform_random = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        net = tf.concat(1, [state, theta_phi, last_theta_phi])
        net = slim.fully_connected(net, 500, weights_initializer=fanin_init(net), biases_initializer=fanin_init(net))
        net = slim.fully_connected(net, 1, weights_initializer=uniform_random, biases_initializer=uniform_random,
                                   activation_fn=None)

        return tf.squeeze(net, [1])

state_placeholder = tf.placeholder(tf.float32, [None, STATE_DIM], 'state')
next_state_placeholder = tf.placeholder(tf.float32, [None, STATE_DIM], 'next_state')
state_placeholder = 2.0 * (state_placeholder - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) - 1
next_state_placeholder = 2.0 * (next_state_placeholder - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) - 1

temporally_extended_states_placeholder = tf.placeholder(tf.float32, [None, TIME_DIM, STATE_DIM], 'temporally_extended_states')
theta_phi_placeholder = tf.placeholder(tf.float32, [None, FLATTENED_THETA_PHI_DIM], 'theta_phi')
last_theta_phi_placeholder = tf.placeholder(tf.float32, [None, FLATTENED_THETA_PHI_DIM], 'last_theta_phi')
future_theta_phi_placeholder = tf.placeholder(tf.float32, [None, FLATTENED_THETA_PHI_DIM], 'future_theta_phi')
reward_placeholder = tf.placeholder(tf.float32, [None], 'reward')

done_placeholder = tf.placeholder(tf.bool, [None], 'done')

train_actor_output, (_, train_actor_output_residual) = ActorCell(outer_scope='train_network')(state_placeholder, state=(last_theta_phi_placeholder, None))
train_actor_next_output, (_, train_actor_next_output_residual) = ActorCell(outer_scope='train_network', reuse=True)(next_state_placeholder, state=(phishift(train_actor_output), None))
target_actor_output, (_, target_actor_output_residual) = ActorCell(outer_scope='target_network')(state_placeholder, state=(last_theta_phi_placeholder, None))
target_actor_next_output, (_, target_actor_next_output_residual) = ActorCell(outer_scope='target_network', reuse=True)(next_state_placeholder, state=(tf.stop_gradient(phishift(target_actor_output)), None))

# Expose functions to environment loop
phi_from_theta_phi_placeholder = phi(theta_phi_placeholder)
phishift_from_theta_phi_placeholder = phishift(theta_phi_placeholder)

target_critic_next_output = critic_network(next_state_placeholder, target_actor_next_output, phishift(target_actor_output), outer_scope='target_network')
train_critic_current_action = critic_network(state_placeholder, train_actor_output, last_theta_phi_placeholder, outer_scope='train_network')
train_critic_placeholder_action = critic_network(state_placeholder, theta_phi_placeholder, last_theta_phi_placeholder, outer_scope='train_network', reuse=True)

train_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/actor')
target_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network/actor')
train_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/critic')
target_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network/critic')

with tf.name_scope('actor_loss'):
    weight_decay_actor = tf.add_n([ACTOR_L2_WEIGHT_DECAY * tf.reduce_sum(var**2) for var in train_actor_vars])
    optim_actor = tf.train.AdamOptimizer(ACTOR_LR)
    loss_actor = -tf.reduce_mean(train_critic_current_action) + weight_decay_actor

    # Actor Optimization
    grads_and_vars_actor = optim_actor.compute_gradients(loss_actor, var_list=train_actor_vars)
    optimize_actor = optim_actor.apply_gradients(grads_and_vars_actor)

    with tf.control_dependencies([optimize_actor]):
        train_target_vars = zip(train_actor_vars, target_actor_vars)
        train_actor = tf.group(*[target.assign(TAU * train + (1 - TAU) * target) for train, target in train_target_vars])

with tf.name_scope('critic_loss'):
    combined_reward = reward_placeholder - BETA * tf.reduce_sum(tf.abs(phishift(theta_phi_placeholder) - last_theta_phi_placeholder), axis=1)
    q_target_value = tf.select(done_placeholder, combined_reward, combined_reward + GAMMA * tf.stop_gradient(target_critic_next_output))
    q_error = (tf.abs(q_target_value - train_critic_placeholder_action) + 1)**2
    q_error_batch = tf.reduce_mean(q_error)
    weight_decay_critic = tf.add_n([CRITIC_L2_WEIGHT_DECAY * tf.reduce_sum(var**2) for var in train_critic_vars])
    loss_critic = q_error_batch + weight_decay_critic

    # Critic Optimization
    optim_critic = tf.train.AdamOptimizer(CRITIC_LR)
    grads_and_vars_critic = optim_critic.compute_gradients(loss_critic, var_list=train_critic_vars)
    optimize_critic = optim_critic.apply_gradients(grads_and_vars_critic)
    with tf.control_dependencies([optimize_critic]):
        train_target_vars = zip(train_critic_vars, target_critic_vars)
        train_critic = tf.group(*[target.assign(TAU * train + (1 - TAU) * target) for train, target in train_target_vars])

sess = tf.Session()
writer = tf.summary.FileWriter("./.tflogs", sess.graph)
sess.run(tf.global_variables_initializer())

# Initialize target = train
sess.run([target.assign(train) for train, target in zip(train_actor_vars, target_actor_vars)])
sess.run([target.assign(train) for train, target in zip(train_critic_vars, target_critic_vars)])

sess.graph.finalize()

replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
replay_priorities = np.zeros(replay_buffer.maxlen, dtype=np.float64)
replay_priorities_sum = 0
rewards = []

easy_shift = lambda x: sess.run(phishift_from_theta_phi_placeholder, feed_dict={theta_phi_placeholder: [x]})[0]

for episode in tqdm(xrange(1000)):
    env_state = env.reset()
    eta_noise = Noise()
    last_theta_phi = np.zeros(FLATTENED_THETA_PHI_DIM)
    training = len(replay_buffer) - TIME_DIM >= min(ITERATIONS_BEFORE_TRAINING, replay_buffer.maxlen)
    testing = (episode + 1) % 2 == 0 and training
    history = []

    for step in tqdm(xrange(MAX_EPISODE_LENGTH)):
        theta_phi = sess.run(train_actor_output, feed_dict={state_placeholder: [env_state], last_theta_phi_placeholder: [last_theta_phi]})[0]

        theta_phi = theta_phi if testing else theta_phi + np.clip(abs(eta_noise.perlin(step=step)), 0, 1) * np.array([0, 1, 1, 0])

        print theta_phi

        env.render()

        action = sess.run(phi_from_theta_phi_placeholder, feed_dict={theta_phi_placeholder: [theta_phi]})

        assert action.shape == env.action_space.sample().shape, (action.shape, env.action_space.sample().shape)

        env_next_state, env_reward, env_done, env_info = env.step(np.clip(action, -1, 1) * 2)

        history.append((theta_phi, action, env_next_state))
        replay_buffer.append([env_state, theta_phi, env_reward, env_next_state, env_done, last_theta_phi])

        replay_priorities_sum -= replay_priorities[len(replay_buffer) - 1]
        replay_priorities[len(replay_buffer) - 1] = 300
        replay_priorities_sum += replay_priorities[len(replay_buffer) - 1]

        env_state = env_next_state
        last_theta_phi = easy_shift(theta_phi)

        if training:
            p_errors = replay_priorities[:len(replay_buffer) - TIME_DIM] / (replay_priorities_sum - replay_priorities[len(replay_buffer) - TIME_DIM:len(replay_buffer)].sum())
            minibatch_indexes = np.random.choice(xrange(len(replay_buffer) - TIME_DIM), size=BATCH_SIZE, replace=False, p=p_errors)

            state_batch = [replay_buffer[i][0] for i in minibatch_indexes]
            theta_phi_batch = [replay_buffer[i][1] for i in minibatch_indexes]
            reward_batch = [replay_buffer[i][2] for i in minibatch_indexes]
            next_state_batch = [replay_buffer[i][3] for i in minibatch_indexes]
            done_batch = [replay_buffer[i][4] for i in minibatch_indexes]
            last_theta_phi_batch = [replay_buffer[i][5] for i in minibatch_indexes]

            temporally_extended_states_batch = [[replay_buffer[i + t][0] for t in range(TIME_DIM)] for i in minibatch_indexes]
            future_theta_phi = [replay_buffer[i + FUTURE_THETA_N][1] if len(replay_buffer) > i + FUTURE_THETA_N else replay_buffer[i][1] for i in minibatch_indexes]

            _, _, errors, l = sess.run([train_actor, train_critic, q_error, q_error_batch],
                                            feed_dict={
                                                state_placeholder: state_batch,
                                                theta_phi_placeholder: theta_phi_batch,
                                                last_theta_phi_placeholder: last_theta_phi_batch,
                                                reward_placeholder: reward_batch,
                                                next_state_placeholder: next_state_batch,
                                                future_theta_phi_placeholder: future_theta_phi,
                                                done_placeholder: done_batch,
                                                temporally_extended_states_placeholder: temporally_extended_states_batch
                                            })

            # img = np.reshape(np.hstack([np.ndarray.flatten(np.array([v])) for v in sess.run(train_actor_vars)])[:3000], [50, 60])
            # img /= img.max()
            # img2 = np.reshape(np.hstack([np.ndarray.flatten(np.array([v])) for v in sess.run(train_critic_vars)])[:3000], [50, 60])
            # img2 /= img.max()
            # cv2.imshow("criticactor", np.vstack([img, img2]))

            for i, error in zip(minibatch_indexes, errors):
                replay_priorities_sum -= replay_priorities[i]
                replay_priorities[i] = error
                replay_priorities_sum += replay_priorities[i]

        if env_done:
            break

    reward_bonus_normalizer = sum(np.abs(b[1][0] - b[5][0]) != 0 for b in [replay_buffer[i] for i in range(len(replay_buffer) - step - 1, len(replay_buffer))])

    print 'Is Testing:', testing
    print 'Total Penalty: ', reward_bonus_normalizer
    print '\n'

    # data = []
    # tmp_phi = theta_phi
    # for i in range(1000):
    #     data.append(history[i][1])
    #
    # plt.plot(data)
    # plt.show()

    # if episode >= 10 and testing:
    #     plt.setp(plt.plot([x[1][0] for x in history], 'o'), 'color', 'black', 'linewidth', 3.0)
    #     #plt.setp(plt.plot([x[2][0] - 4 for x in history]), 'color', 'red', 'linewidth', 3.0)
    #     for i in range(0, len(history)):
    #         # split_parameters = np.split(np.reshape(history[i][0], [-1, ACTION_DIM, RBF_NUM_PARAMETERS, THETA_PHI_PER_ACTION_DIM // RBF_NUM_PARAMETERS]), RBF_NUM_PARAMETERS, axis=2)
    #         # weights, centers, sigmas = [a * b for a, b in zip(split_parameters, RBF_MULTIPLIERS)]
    #
    #         subdata = []
    #         xs = np.linspace(i, i+10, 10)
    #         for x in xs:
    #             subdata.append(history[i][0][0])
    #             #subdata.append(np.tanh(np.sum(weights * np.exp(-(np.abs(x - centers - i) ** 2) / (2 * sigmas ** 2)))))
    #
    #         plt.plot(xs, subdata, 'ro', markersize=4, fillstyle='full', markeredgecolor='red', markeredgewidth=0.0)
    #
    #
    #     plt.show()

    # h = np.hstack([weights, sigmas, centers])
    # a = h.T
    # a = a - a.min()
    # a /= a.max()
    # b = np.hstack([np.zeros([a.shape[0], 1]), np.diff(a, n=1, axis=1)])
    # plt.imshow(np.vstack([a, np.abs(b)]), interpolation='nearest', aspect='auto')
    # plt.colorbar()
    # plt.show()