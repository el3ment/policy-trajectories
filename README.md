# Result from beta=3, spark=off
# Episode 11
# Is Testing: True
# Total Reward:  -0.445748907633
# Total Penalty:  1000 11.048982956

# Result from beta=3, spark=yes
# Episode 11
# Total Reward:  -241.013091761
# Total Penalty:  275 188.191666067

# Result from beta=0, spark=off
# Episode 11
# Total Reward:  -263.696924855
# Total Penalty:  1000 16.4907716438

# Result from beta=0, spark=yes
# Episode 11
# Total Reward:  -122.242187667
# Total Penalty:  414 268.761390388

test


# def phi(theta, t=0):
#     split_parameters = tf.split(2, TRAJECTORY_MODEL_NUM_PARAMETERS, tf.reshape(theta, [-1, ACTION_DIM,
#                                                                                        TRAJECTORY_MODEL_NUM_PARAMETERS,
#                                                                                        THETA_PHI_PER_ACTION_DIM // TRAJECTORY_MODEL_NUM_PARAMETERS]))
#     split_parameters = [a * b for a, b in
#                         zip(split_parameters, TRAJECTORY_MODEL_MULTIPLIERS)]  # weights, centers, sigmas
#     return tf.tanh(tf.reduce_sum(split_parameters[0] * tf.exp(
#         -((t - split_parameters[1]) ** 2) / (2 * tf.clip_by_value(split_parameters[2] + 1, 0.01, 10) ** 2)),
#                                  reduction_indices=[1, 2, 3]))
#
# def phishift(theta):
#     split_parameters = tf.split(2, TRAJECTORY_MODEL_NUM_PARAMETERS, tf.reshape(theta, [-1, ACTION_DIM,
#                                                                                        TRAJECTORY_MODEL_NUM_PARAMETERS,
#                                                                                        THETA_PHI_PER_ACTION_DIM // TRAJECTORY_MODEL_NUM_PARAMETERS]))
#     split_parameters[1] = (split_parameters[1] * TRAJECTORY_MODEL_MULTIPLIERS[1] - 1) / float(
#         TRAJECTORY_MODEL_MULTIPLIERS[1])
#     return tf.reshape(tf.concat(2, split_parameters), [-1, ACTION_DIM * THETA_PHI_PER_ACTION_DIM])



# def phi(theta, t=0):
#     return tf.reduce_sum(theta, reduction_indices=[1])
#
#
# def phishift(theta):
#     return tf.identity(theta)