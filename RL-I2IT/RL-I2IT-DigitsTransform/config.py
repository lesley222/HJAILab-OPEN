
class Config:

    WIDTH = 28
    HEIGHT = 28
    STATE_CHANNEL = 2
    ENTROPY_BETA = 0.001
    SEED = 100
    GPU_ID = 0

    # train
    MAX_EP_STEPS = 100
    EP_BOOTSTRAP = 200
    MAX_GLOBAL_EP = 50001
    UPDATE_GLOBAL_ITER = 10
    SAMPLE_BATCH_SIZE = 32
    FREQUENCY_ACTOR = 2
    FREQUENCY_VAE = 1
    FREQUENCY_SAVE_MODEL = 5000


    # main parameters
    NF = 32
    DIM_Z = 49
    TAU = 0.005
    GAMMA = 0.99
    LEARNING_RATE = 3e-5
    SCORE_THRESHOLD = 0.96
    MEMORY_SIZE = 20000


    # MNIST_DIR = 'data'
    MNIST_DIR = 'data\\MNIST\\raw' # windows path
    LOG_DIR = 'log/'
    MODEL_PATH = 'model/'
    PROCESS_PATH = 'process'
    TEST_PATH = 'result'

    idx = 50000
    ACTOR_MODEL = MODEL_PATH + 'actor_{}.ckpt'.format(idx)
    CRITIC1_MODEL = MODEL_PATH + 'critic1_{}.ckpt'.format(idx)
    CRITIC2_MODEL = MODEL_PATH + 'critic2_{}.ckpt'.format(idx)
    DECODER_MODEL_RL = MODEL_PATH + 'decoder_{}.ckpt'.format(idx)



