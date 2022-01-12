import datetime
ENV_NAME = 'BipedalWalker-v3'
SEED = 9
#7 000 000 STEPS / batch_size
NUMBER_OF_STEPS = 3500
NUMBER_OF_EPISODES = 100
BATCH_SIZE = 2048
MINIBATCH_SIZE = 64
UPDATE_STEPS = 10

GAE = True
GAMMA = 0.99
LAMBDA = 0.97

CLIPPING_EPSILON = 0.2
LEARNING_RATE_POLICY = 0.0004
LEARNING_RATE_CRITIC = 0.001
MAX_GRAD_NORM = 0.5


WRITER_FLAG = True
now = datetime.datetime.now()
date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
gae = 'gae' if GAE else ''
WRITER_NAME = 'PPO_'+ENV_NAME+'_'+str(LEARNING_RATE_POLICY)+'_'+str(LEARNING_RATE_CRITIC)+'_'+str(BATCH_SIZE)+'_'+str(MINIBATCH_SIZE)+'_'+str(SEED)+'-norm_'+gae+str(LAMBDA)+'-'+date_time

