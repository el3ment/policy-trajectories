import gym
from tqdm import tqdm
import tensorflow as tf
from collections import deque
import tensorflow.contrib.slim as slim
import random
import numpy as np
from scipy import stats

env = gym.make('Pendulum-v0')

TAU = 0.001
GAMMA = .99
EPSILON = .99
ACTOR_LR = 0.001
CRITIC_LR = 0.0001
ACTOR_L2_WEIGHT_DECAY = 0.0
CRITIC_L2_WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
ITERATIONS_BEFORE_TRAINING = BATCH_SIZE + 1
REPLAY_BUFFER_SIZE = 10000

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


def fanin_init(layer):
    fanin = layer.get_shape().as_list()[1]
    v = 1.0 / np.sqrt(fanin)
    return tf.random_uniform_initializer(minval=-v, maxval=v)


class Noise:
    def __init__(self,):
        global action_dim
        self.state = np.zeros(action_dim)

    def josh(self, mu, sigma):
        return stats.truncnorm.rvs((-1 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma, size=len(self.state))

    def ou(self, theta, sigma):
        self.state -= theta * self.state - sigma * np.random.randn(len(self.state))
        return self.state


def actor_network(state, outer_scope, reuse=False):
    global action_dim
    with tf.variable_scope(outer_scope + '/actor', reuse=reuse):
        uniform_random = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        net = slim.fully_connected(state, 1024, weights_initializer=fanin_init(state), biases_initializer=fanin_init(state))
        net = slim.fully_connected(net, 512, weights_initializer=fanin_init(net), biases_initializer=fanin_init(net))
        return slim.fully_connected(net, action_dim, weights_initializer=uniform_random, biases_initializer=uniform_random, activation_fn=tf.tanh)


def critic_network(state, action, outer_scope, reuse=False):
    with tf.variable_scope(outer_scope + '/critic', reuse=reuse):
        uniform_random = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        net = tf.concat(1, [state, action])
        net = slim.fully_connected(net, 1024, weights_initializer=fanin_init(state), biases_initializer=fanin_init(state))
        net = slim.fully_connected(net, 512, weights_initializer=fanin_init(net), biases_initializer=fanin_init(net))
        net = slim.fully_connected(net, 1, weights_initializer=uniform_random, biases_initializer=uniform_random, activation_fn=None)

        return tf.squeeze(net, [1])


state_placeholder = tf.placeholder(tf.float32, [None, state_dim], 'state')
action_placeholder = tf.placeholder(tf.float32, [None, action_dim], 'action_train')
reward_placeholder = tf.placeholder(tf.float32, [None], 'reward')
next_state_placeholder = tf.placeholder(tf.float32, [None, state_dim], 'next_state')
done_placeholder = tf.placeholder(tf.bool, [None], 'done')

train_actor_output = actor_network(state_placeholder, outer_scope='train_network')
target_actor_next_output = actor_network(next_state_placeholder, outer_scope='target_network')

target_critic_next_output = critic_network(next_state_placeholder, target_actor_next_output, outer_scope='target_network')
train_critic_current_action = critic_network(state_placeholder, train_actor_output, outer_scope='train_network')
train_critic_placeholder_action = critic_network(state_placeholder, action_placeholder, outer_scope='train_network', reuse=True)

train_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/actor')
target_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network/actor')
train_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/critic')
target_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network/critic')

with tf.name_scope('actor_loss'):
    weight_decay_actor = tf.add_n([ACTOR_L2_WEIGHT_DECAY * tf.nn.l2_loss(var) for var in train_actor_vars])
    loss_actor = -tf.reduce_mean(train_critic_current_action) + weight_decay_actor

    # Actor Optimization
    optim_actor = tf.train.AdamOptimizer(ACTOR_LR)
    grads_and_vars_actor = optim_actor.compute_gradients(loss_actor, var_list=train_actor_vars)

    optimize_actor = optim_actor.apply_gradients(grads_and_vars_actor)
    with tf.control_dependencies([optimize_actor]):
        train_target_vars = zip(train_actor_vars, target_actor_vars)
        train_actor = tf.group(*[target.assign(TAU * train + (1 - TAU) * target) for train, target in train_target_vars])

with tf.name_scope('critic_loss'):
    q_target_value = tf.stop_gradient(tf.select(done_placeholder, reward_placeholder, reward_placeholder + GAMMA * target_critic_next_output))
    q_error = (q_target_value - train_critic_placeholder_action)**2
    q_error_batch = tf.reduce_mean(q_error)
    weight_decay_critic = tf.add_n([CRITIC_L2_WEIGHT_DECAY * tf.nn.l2_loss(var) for var in train_critic_vars])
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
sess.run(tf.initialize_all_variables())

# Initialize target = train
sess.run([target.assign(train) for train, target in zip(train_actor_vars, target_actor_vars)])
sess.run([target.assign(train) for train, target in zip(train_critic_vars, target_critic_vars)])

replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
replay_errors = np.zeros(replay_buffer.maxlen, dtype=np.float64)
replay_error_sum = 0
rewards = []

for episode in tqdm(xrange(1000)):
    env_state = env.reset()
    eta_noise = Noise()
    training = len(replay_buffer) >= min(ITERATIONS_BEFORE_TRAINING, replay_buffer.maxlen)
    testing = episode % 2 == 0 and training

    for step in tqdm(xrange(1000)):
        action = sess.run(train_actor_output, feed_dict={state_placeholder: [env_state]})[0]
        action += 0 if testing else eta_noise.ou(theta=.15, sigma=.2)
        # action = action if testing else eta_noise.josh(action, sigma=10*(EPSILON**(episode+(step/2.0))))

        env_next_state, env_reward, env_done, env_info = env.step(np.clip(action, -1, 1) * 2)

        if not testing:
            replay_buffer.append((env_state, action, env_reward, env_next_state, env_done))

            replay_error_sum -= replay_errors[len(replay_buffer) - 1]
            replay_errors[len(replay_buffer) - 1] = 100.0
            replay_error_sum += 100.0

        env_state = env_next_state

        if training:
            env.render()
            p_errors = replay_errors[:len(replay_buffer)] / replay_error_sum
            minibatch_indexes = np.random.choice(xrange(len(replay_buffer)), size=BATCH_SIZE, replace=False, p=p_errors)

            state_batch = [replay_buffer[i][0] for i in minibatch_indexes]
            action_batch = [replay_buffer[i][1] for i in minibatch_indexes]
            reward_batch = [replay_buffer[i][2] for i in minibatch_indexes]
            next_state_batch = [replay_buffer[i][3] for i in minibatch_indexes]
            done_batch = [replay_buffer[i][4] for i in minibatch_indexes]

            _, _, errors, lc, la = sess.run([train_actor, train_critic, q_error, loss_critic, loss_actor], feed_dict={
                state_placeholder: state_batch,
                action_placeholder: action_batch,
                reward_placeholder: reward_batch,
                next_state_placeholder: next_state_batch,
                done_placeholder: done_batch
            })

            for i, error in zip(minibatch_indexes, errors):
                replay_error_sum -= replay_errors[i]
                replay_errors[i] = error
                replay_error_sum += replay_errors[i]

        if env_done:
            break
