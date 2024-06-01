class Config:
    # Global Variables
    NAME = 'color-lines_style'
    SCORE_THRESHOLD = 14.
    IMAGE_SIZE = 256
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    STYLE_WEIGHT = 1e5
    CONTENT_WEIGHT = 1e0
    TV_WEIGHT = 1e-7
    SEED = 100

    GAMMA = 0.99
    MEMORY_SIZE = 3000
    MAX_GLOBAL_EPISODES = 400001

    FREQUENCY_ACTOR = 2
    FREQUENCY_VAE = 1
    UPDATE_GLOBAL_ITER = 4
    EP_BOOTSTRAP = 10
    SAMPLE_BATCH_SIZE = 1

    FIRST_EP = 20
    MAX_EP_STEPS = 10

    MODEL_PATH = './model/'
    RESULT_PATH = './result/'

    DATASET_ROOT = '../../../Datasets/MS-COCO-2014/train2014'
    # DATASET_ROOT = '../../../datasets/train2014'
    TEST_DATASET_ROOT = '../../../Datasets/MS-COCO-2014/test2014'
    STYLE_PATH = './datasets/styles/blue_swirls.jpg'
    LOG_DIR = './logs/'
    CURVE_DIR = 'curve'

    GPU_ID = 0  # 1, 2, -1-
    TEST_GPU_ID = 4

    encoder_model_path = './model/actor_blue_swirls_style.ckpt'
    decoder_model_path = './model/decoder_blue_swirls_style.ckpt'

