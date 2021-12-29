
ENV_NAME = 'BipedalWalker-v3'
NUMBER_OF_STEPS = 156
TRAJECTORY_LENGTH = 100
MINIBATCH_SIZE = 25
GAMMA = 0.99

HYPERPARAMETERS = {
    'learning_rate': 0.003,
    'gamma': 0.99,
    'random_seed': 12,
    'baseline': True,
    'test_counter': 8,
    'env_name': 'CartPole-v1',
    'writer_test': True,
    'writer_train': False,
    'writer_log_dir': 'content/runs/REINFORCE-3232-3-baseline-seed=-1',
    'max_train_games': 5000,
    'max_test_games': 10,
    'print_test_results': True
}
