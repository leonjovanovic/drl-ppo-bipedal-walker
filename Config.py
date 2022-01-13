import datetime

ENV_NAME = 'BipedalWalker-v3'
SEED = 9
# 7 000 000 STEPS / batch_size
NUMBER_OF_STEPS = 1000
NUMBER_OF_EPISODES = 100
BATCH_SIZE = 2048
MINIBATCH_SIZE = 32
UPDATE_STEPS = 10

GAE = True
GAMMA = 0.99
LAMBDA = 0.97

CLIPPING_EPSILON = 0.2
ANNEAL_LR = True
LEARNING_RATE_POLICY = 0.0004
LEARNING_RATE_CRITIC = 0.0005
MAX_GRAD_NORM = 0.5

ENTROPY_COEF = 0
ENV_SCALE_CROP = True

WRITER_FLAG = True
now = datetime.datetime.now()
date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
gae = 'gae' if GAE else ''
env = 'env_scaled' if ENV_SCALE_CROP else ''
WRITER_NAME = 'PPO_' + ENV_NAME + '_' + str(LEARNING_RATE_POLICY) + '_' + str(LEARNING_RATE_CRITIC) + '_' + str(
    BATCH_SIZE) + '_' + str(MINIBATCH_SIZE) + '_' + str(SEED) + '-norm_' + gae + str(LAMBDA) + '-' + str(
    ENTROPY_COEF) + '-' + env + '-' + date_time
