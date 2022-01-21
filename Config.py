import datetime

#Izbrisati nepotrebno u main i testu
#eps_start_end
#nauciti sta radi LAMBDA u GAE
#nauciti sta radi EPSILON u Adam

ENV_NAME = 'BipedalWalker-v3'
SEED = 9
NUMBER_OF_STEPS = 600 # probati 500, 600, 700, 800, ... 1500
NUMBER_OF_EPISODES = 100
BATCH_SIZE = 2048
MINIBATCH_SIZE = 32
UPDATE_STEPS = 10

GAE = True
GAMMA = 0.99
LAMBDA = 0.8

CLIPPING_EPSILON = 0.2
LEARNING_RATE_POLICY = 0.0003
LEARNING_RATE_CRITIC = 0.0004
ANNEAL_LR = True
MAX_GRAD_NORM = 0.5
EPSILON_START = 1e-5 # napraviti u Agentu da pada sa 1e-5 do 1e-6 kako raste NUMBER_OF_STEPS
EPSILON_END = 1e-5

ENTROPY_COEF = 0
ENV_SCALE_CROP = True

WRITER_FLAG = True

# ---------------------------------------- Output -----------------------------------------------
gae = 'gae' + str(LAMBDA) if GAE else ''
anneal = 'decay' if ANNEAL_LR else ''
env = 'env_scaled' if ENV_SCALE_CROP else ''
now = datetime.datetime.now()
date_time = "{}.{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
WRITER_NAME = 'PPO_BW-v3' + '_' + str(SEED) + "_" + str(NUMBER_OF_STEPS) + "_" + str(BATCH_SIZE) + "_" + \
              str(MINIBATCH_SIZE) + "_" + str(UPDATE_STEPS) + "_" + gae + "_" + str(GAMMA) + "_" + \
              str(CLIPPING_EPSILON) + "_" + str(LEARNING_RATE_POLICY)[-2:] + "_" + str(LEARNING_RATE_CRITIC)[-2:] + "_" + \
              anneal + "_" + str(EPSILON_START)[-2:] + "_" + str(EPSILON_END)[-2:] + "_" + env + "_" + str(ENTROPY_COEF) + '_' + date_time
