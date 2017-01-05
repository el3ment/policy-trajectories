import gym

env = gym.make('Pendulum-v0')
env.reset(), env.render()  # HACK - bug with rendering if called after tensorflow import :( just started

from tqdm import tqdm
import tensorflow as tf
from collections import deque
import tensorflow.contrib.slim as slim
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

FUTURE_THETA_N = 1
TIME_DIM = 1

TAU = 0.01
GAMMA = .99
ACTOR_LR = 0.001
CRITIC_LR = 0.001
ACTOR_L2_WEIGHT_DECAY = 0.00
CRITIC_L2_WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50000
MAX_EPISODE_LENGTH = 1000
ITERATIONS_BEFORE_TRAINING = BATCH_SIZE + 1 + TIME_DIM
BETA = 0

RBF_NUM_KERNELS = 1
RBF_NUM_PARAMETERS = 1
RBF_MULTIPLIERS = [1, 5, 2]  # weights, centers, sigmas

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
THETA_PHI_PER_ACTION_DIM = RBF_NUM_KERNELS * RBF_NUM_PARAMETERS
FLATTENED_THETA_PHI_DIM = THETA_PHI_PER_ACTION_DIM * ACTION_DIM

# Inverting Gradients for Bounded Control - https://www.cs.utexas.edu/~AustinVilla/papers/ICLR16-hausknecht.pdf


def fanin_init(layer):
    fanin = layer.get_shape().as_list()[1]
    v = 1.0 / np.sqrt(fanin)
    return tf.random_uniform_initializer(minval=-v, maxval=v)


class Noise:
    def __init__(self, ):
        global FLATTENED_THETA_PHI_DIM
        self.state = np.zeros(FLATTENED_THETA_PHI_DIM)

    def josh(self, mu, sigma):
        return stats.truncnorm.rvs((-1 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma, size=len(self.state))

    def ou(self, theta, sigma):
        self.state -= theta * self.state - sigma * np.random.randn(len(self.state))
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
            net = tf.concat(1, [inputs, last_theta_phi])
            net = slim.fully_connected(net, 1024, weights_initializer=fanin_init(net),
                                       biases_initializer=fanin_init(net))
            net = slim.fully_connected(net, 512, weights_initializer=fanin_init(net),
                                       biases_initializer=fanin_init(net))
            residual = slim.fully_connected(net, FLATTENED_THETA_PHI_DIM, weights_initializer=uniform_random,
                                            biases_initializer=uniform_random, activation_fn=tf.tanh)
            output = tf.clip_by_value(residual + last_theta_phi, -1, 1)

        return output, (output, residual)


def critic_network(state, theta_phi, last_theta_phi, outer_scope, reuse=False):
    with tf.variable_scope(outer_scope + '/critic', reuse=reuse):
        uniform_random = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        net = tf.concat(1, [state, theta_phi, last_theta_phi])
        net = slim.fully_connected(net, 1024, weights_initializer=fanin_init(net),
                                   biases_initializer=fanin_init(net))
        net = slim.fully_connected(net, 512, weights_initializer=fanin_init(net),
                                   biases_initializer=fanin_init(net))
        net = slim.fully_connected(net, 1, weights_initializer=uniform_random, biases_initializer=uniform_random,
                                   activation_fn=None)

        return tf.squeeze(net, [1])


# def phi(theta, t=0):
#     split_parameters = tf.split(2, RBF_NUM_PARAMETERS, tf.reshape(theta, [-1, ACTION_DIM, RBF_NUM_PARAMETERS, THETA_PHI_PER_ACTION_DIM // RBF_NUM_PARAMETERS]))
#     split_parameters = [a * b for a, b in zip(split_parameters, RBF_MULTIPLIERS)]
#     return tf.tanh(tf.reduce_sum(split_parameters[0] * tf.exp(-((t - split_parameters[1]) ** 2) / (2 * split_parameters[2] ** 2)), reduction_indices=[1, 2, 3]))
#
#
# def phishift(theta):
#     split_parameters = tf.split(2, RBF_NUM_PARAMETERS, tf.reshape(theta, [-1, ACTION_DIM, RBF_NUM_PARAMETERS, THETA_PHI_PER_ACTION_DIM // RBF_NUM_PARAMETERS]))
#     split_parameters[1] = (split_parameters[1] * RBF_MULTIPLIERS[1] - 1) / float(RBF_MULTIPLIERS[1])
#     return tf.reshape(tf.concat(2, split_parameters), [-1, ACTION_DIM * THETA_PHI_PER_ACTION_DIM])


def phi(theta, t=0):
    return tf.reduce_sum(theta, reduction_indices=[1])


def phishift(theta):
    return tf.identity(theta)


state_placeholder = tf.placeholder(tf.float32, [None, STATE_DIM], 'state')
temporally_extended_states_placeholder = tf.placeholder(tf.float32, [None, TIME_DIM, STATE_DIM], 'temporally_extended_states')
theta_phi_placeholder = tf.placeholder(tf.float32, [None, FLATTENED_THETA_PHI_DIM], 'theta_phi')
last_theta_phi_placeholder = tf.placeholder(tf.float32, [None, FLATTENED_THETA_PHI_DIM], 'last_theta_phi')
future_theta_phi_placeholder = tf.placeholder(tf.float32, [None, FLATTENED_THETA_PHI_DIM], 'future_theta_phi')
reward_placeholder = tf.placeholder(tf.float32, [None], 'reward')
reward_bonus_placeholder = tf.placeholder(tf.float32, [None], 'reward_bonus')
next_state_placeholder = tf.placeholder(tf.float32, [None, STATE_DIM], 'next_state')
done_placeholder = tf.placeholder(tf.bool, [None], 'done')

train_actor_output, (_, train_actor_output_residual) = ActorCell(outer_scope='train_network')(state_placeholder, state=(last_theta_phi_placeholder, None))
train_actor_next_output, (_, train_actor_next_output_residual) = ActorCell(outer_scope='train_network', reuse=True)(next_state_placeholder, state=(train_actor_output, None))
target_actor_output, (_, target_actor_output_residual) = ActorCell(outer_scope='target_network')(state_placeholder, state=(last_theta_phi_placeholder, None))
target_actor_next_output, (_, target_actor_next_output_residual) = ActorCell(outer_scope='target_network', reuse=True)(next_state_placeholder, state=(tf.stop_gradient(target_actor_output), None))

# Policy Roll-out
# train_actor_cell = ActorCell(outer_scope='train_network', reuse=True)
# state = last_theta_phi_placeholder
# extended_train_actor_residuals_list, extended_train_actor_outputs_list = [], []
# for input in tf.unstack(temporally_extended_states_placeholder, axis=1):
#     output, (state, residual) = train_actor_cell(input, (state, None))
#     extended_train_actor_residuals_list.append(residual)
#     extended_train_actor_outputs_list.append(output)
# extended_train_actor_residuals = tf.stack(extended_train_actor_residuals_list, axis=1)

# Expose functions to environment loop
phi_from_theta_phi_placeholder = phi(theta_phi_placeholder)
phishift_from_theta_phi_placeholder = phishift(theta_phi_placeholder)

target_critic_next_output = critic_network(next_state_placeholder, target_actor_next_output, target_actor_output, outer_scope='target_network')
train_critic_current_action = critic_network(state_placeholder, train_actor_output, last_theta_phi_placeholder, outer_scope='train_network')
train_critic_placeholder_action = critic_network(state_placeholder, theta_phi_placeholder, last_theta_phi_placeholder, outer_scope='train_network', reuse=True)

train_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/actor')
target_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network/actor')
train_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/critic')
target_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network/critic')

with tf.name_scope('actor_loss'):
    weight_decay_actor = tf.add_n([ACTOR_L2_WEIGHT_DECAY * tf.reduce_sum(var**2) for var in train_actor_vars])

    optim_actor = tf.train.AdamOptimizer(ACTOR_LR)

    # mask = tf.sign(tf.reduce_max(tf.abs(extended_train_actor_residuals), axis=1, keep_dims=True) - tf.abs(extended_train_actor_residuals))
    # loss_sparsity = tf.reduce_mean(mask * tf.abs(extended_train_actor_residuals))

    # loss_sparsity = tf.reduce_sum(tf.abs(extended_train_actor_residuals)**0.01, axis=[1])**(1.0/0.01)
    # loss_sparsity = tf.reduce_mean(loss_sparsity, axis=[0])

    # sparsity_rollout = tf.stack([phi(extended_train_actor_outputs_list[0], t=t) for t in range(TIME_DIM)], axis=1)
    # actual_rollout = tf.stack([phi(tp) for tp in extended_train_actor_outputs_list], axis=1)
    # loss_sparsity = tf.reduce_mean((sparsity_rollout - actual_rollout)**2)

    loss_actor = -tf.reduce_mean(train_critic_current_action) + weight_decay_actor

    # Actor Optimization
    grads_and_vars_actor = optim_actor.compute_gradients(loss_actor, var_list=train_actor_vars)
    optimize_actor = optim_actor.apply_gradients(grads_and_vars_actor)

    with tf.control_dependencies([optimize_actor]):
        train_target_vars = zip(train_actor_vars, target_actor_vars)
        train_actor = tf.group(*[target.assign(TAU * train + (1 - TAU) * target) for train, target in train_target_vars])

with tf.name_scope('critic_loss'):
    combined_reward = reward_placeholder  # - BETA * tf.stop_gradient(tf.reduce_sum(tf.abs(train_actor_output - last_theta_phi_placeholder), reduction_indices=[1]))
    q_target_value = tf.stop_gradient(tf.select(done_placeholder, combined_reward, combined_reward + GAMMA * target_critic_next_output))
    q_error = (q_target_value - train_critic_placeholder_action) ** 2
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

sess = tf.InteractiveSession()
writer = tf.train.SummaryWriter("./.tflogs", sess.graph)
sess.run(tf.global_variables_initializer())

# Initialize target = train
sess.run([target.assign(train) for train, target in zip(train_actor_vars, target_actor_vars)])
sess.run([target.assign(train) for train, target in zip(train_critic_vars, target_critic_vars)])

replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
replay_priorities = np.zeros(replay_buffer.maxlen, dtype=np.float64)
replay_priorities_sum = 0
rewards = []

for episode in tqdm(xrange(1000)):
    env_state = env.reset()
    eta_noise = Noise()
    last_theta_phi = np.zeros(FLATTENED_THETA_PHI_DIM)
    training = len(replay_buffer) - TIME_DIM - MAX_EPISODE_LENGTH >= min(ITERATIONS_BEFORE_TRAINING, replay_buffer.maxlen)
    testing = episode % 2 == 0 and training
    total_reward = 0
    history = []

    for step in tqdm(xrange(MAX_EPISODE_LENGTH)):
        theta_phi = sess.run(train_actor_output, feed_dict={state_placeholder: [env_state],
                                                            last_theta_phi_placeholder: [last_theta_phi]})[0]

        theta_phi += 0 if testing else eta_noise.ou(theta=.15, sigma=.2)
        env.render()

        action = sess.run(phi_from_theta_phi_placeholder, feed_dict={theta_phi_placeholder: [theta_phi]})

        assert action.shape == env.action_space.sample().shape, (action.shape, env.action_space.sample().shape)

        env_next_state, env_reward, env_done, env_info = env.step(np.clip(action, -1, 1) * 2)
        total_reward += env_reward

        history.append((theta_phi, action, env_next_state))
        replay_buffer.append([env_state, theta_phi, env_reward, env_next_state, env_done, last_theta_phi, np.sum(theta_phi - last_theta_phi)])

        replay_priorities_sum -= replay_priorities[len(replay_buffer) - 1]
        replay_priorities[len(replay_buffer) - 1] = 300
        replay_priorities_sum += replay_priorities[len(replay_buffer) - 1]

        env_state = env_next_state
        last_theta_phi = sess.run(phishift_from_theta_phi_placeholder, feed_dict={theta_phi_placeholder: [theta_phi]})[0]

        if training:
            p_errors = replay_priorities[:len(replay_buffer) - TIME_DIM] / (replay_priorities_sum - replay_priorities[len(replay_buffer) - TIME_DIM:len(replay_buffer)].sum())
            minibatch_indexes = np.random.choice(xrange(len(replay_buffer) - TIME_DIM), size=BATCH_SIZE, replace=False, p=p_errors)

            state_batch = [replay_buffer[i][0] for i in minibatch_indexes]
            theta_phi_batch = [replay_buffer[i][1] for i in minibatch_indexes]
            reward_batch = [replay_buffer[i][2] for i in minibatch_indexes]
            next_state_batch = [replay_buffer[i][3] for i in minibatch_indexes]
            done_batch = [replay_buffer[i][4] for i in minibatch_indexes]
            last_theta_phi_batch = [replay_buffer[i][5] for i in minibatch_indexes]
            reward_bonus_batch = [replay_buffer[i][6] for i in minibatch_indexes]

            temporally_extended_states_batch = [[replay_buffer[i + t][0] for t in range(TIME_DIM)] for i in minibatch_indexes]
            future_theta_phi = [replay_buffer[i + FUTURE_THETA_N][1] if len(replay_buffer) > i + FUTURE_THETA_N else replay_buffer[i][1] for i in minibatch_indexes]

            _, _, errors, r = sess.run([train_actor, train_critic, q_error, combined_reward],
                                            feed_dict={
                                                state_placeholder: state_batch,
                                                theta_phi_placeholder: theta_phi_batch,
                                                last_theta_phi_placeholder: last_theta_phi_batch,
                                                reward_placeholder: reward_batch,
                                                reward_bonus_placeholder: reward_bonus_batch,
                                                next_state_placeholder: next_state_batch,
                                                future_theta_phi_placeholder: future_theta_phi,
                                                done_placeholder: done_batch,
                                                temporally_extended_states_placeholder: temporally_extended_states_batch
                                            })

            for i, error in zip(minibatch_indexes, errors):
                replay_priorities_sum -= replay_priorities[i]
                replay_priorities[i] = error
                replay_priorities_sum += replay_priorities[i]

        if env_done:
            break

    reward_bonus_normalizer = sum(np.abs(b[6]) for b in [replay_buffer[i] for i in range(len(replay_buffer) - step - 1, len(replay_buffer))])
    for i in range(len(replay_buffer) - step - 1, len(replay_buffer)):
        replay_buffer[i][6] /= reward_bonus_normalizer

    print 'Is Testing:', testing
    print 'Total Reward: ', total_reward
    print 'Total Penalty: ', reward_bonus_normalizer
    print '\n'

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