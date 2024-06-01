
class Config:

    DATASET = ''
    HEIGHT = 128
    STATE_CHANNEL = 6
    BOTTLE = 256
    ENTROPY_BETA = 0.01
    SEED = 100

    # train
    MAX_GLOBAL_EP = 400001
    MAX_EP_STEPS = 10
    EP_BOOTSTRAP = 100
    UPDATE_GLOBAL_ITER = 8
    SAMPLE_BATCH_SIZE = 16
    FREQUENCY_ACTOR = 2
    FREQUENCY_VAE = 1

    FIRST_EP = 2000

    PRE_TRAIN = False

    GPU_ID = 0
    NF = 64
    SCORE_THRESHOLD = 30
    MEMORY_SIZE = 5000
    TAU = 0.005
    FIXED_ALPHA = None # finetune alpha is None
    GAMMA = 0.99
    INIT_TEMPERATURE = 0.1

    # main parameters
    LEARNING_RATE = 2e-4

    LOG_DIR = './log/'
    MODEL_PATH = './model/'
    PROCESS_PATH = 'process'
    RESULT_PATH = 'result'

    NAME = 'edges2shoes'

    # TRAIN_PATH = '../datasets/facades/train'
    # TEST_PATH = '../datasets/facades/test'

    TRAIN_PATH = '../datasets/cityscapes/train'
    TEST_PATH = '../datasets/cityscapes/test'

    idx = 100000
    ACTOR_MODEL = MODEL_PATH + 'actor_{}.ckpt'.format(idx)
    CRITIC1_MODEL = MODEL_PATH + 'critic1_{}.ckpt'.format(idx)
    CRITIC2_MODEL = MODEL_PATH + 'critic2_{}.ckpt'.format(idx)
    DECODER_MODEL_RL = MODEL_PATH + 'decoder_{}.ckpt'.format(idx)
    NETD_MODEL = MODEL_PATH + 'netD_{}.ckpt'.format(idx)



