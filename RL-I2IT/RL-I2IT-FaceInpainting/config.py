
class Config:

    DATASET = 'celeba-hq'
    HOLE_SIZE = 64
    HEIGHT = 128
    STATE_CHANNEL = 3
    BOTTLE = 512
    ENTROPY_BETA = 0.01
    SEED = 100

    # train
    MAX_GLOBAL_EP = 5000000
    MAX_EP_STEPS = 20
    EP_BOOTSTRAP = 2000
    UPDATE_GLOBAL_ITER = 10
    SAMPLE_BATCH_SIZE = 32
    FREQUENCY_ACTOR = 2
    FREQUENCY_VAE = 1

    FIRST_EP = 10000

    PRE_TRAIN = False

    GPU_ID = 0
    NF = 64
    SCORE_THRESHOLD = 26.
    MEMORY_SIZE = 100000
    TAU = 0.005
    GAMMA = 0.99

    # main parameters
    LEARNING_RATE = 2e-4

    LOG_DIR = './log/'
    MODEL_PATH = './model/'
    PROCESS_PATH = 'process'
    TEST_PATH = 'result'

    idx = 699900
    ACTOR_MODEL = MODEL_PATH + 'actor_{}.ckpt'.format(idx)
    CRITIC1_MODEL = MODEL_PATH + 'critic1_{}.ckpt'.format(idx)
    CRITIC2_MODEL = MODEL_PATH + 'critic2_{}.ckpt'.format(idx)
    DECODER_MODEL_RL = MODEL_PATH + 'decoder_{}.ckpt'.format(idx)
    NETD_MODEL = MODEL_PATH + 'netD_{}.ckpt'.format(idx)



