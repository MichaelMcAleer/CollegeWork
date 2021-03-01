# -----------------------------------------------------
# Natural Language Processing
# Assignment 3 - Dialogue System
# Michael McAleer R00143621
# -----------------------------------------------------
import os

# Set to True if using Google Colab
GOOGLE_DRIVE_MODE = False

# Set ROOT_DIR path by environment
if GOOGLE_DRIVE_MODE:
    # Run on Google Colab
    ROOT_DIR = ('/content/drive/My Drive/Colab Notebooks/NLP/Assignment3'
                '/Baseline_ChatBot')
else:
    ROOT_DIR = os.getcwd()

# Corpus parent directory
CORPUS_DIR = '{}/corpora'.format(ROOT_DIR)

# Chatbot configuration and corpus files
CHATBOT_PERSONALITY_DIR = '{}/chatbot_personality'.format(CORPUS_DIR)
CHATBOT_DIRECT_PATH = 'chatbot_direct_questions.txt'
CHATBOT_PERSONALITY_PATH = 'chatbot_personality.txt'
CHATBOT_EXIT_PATH = 'chatbot_exit_messages.txt'
CHATBOT_UNKNOWN_RESPONSE_PATH = 'chatbot_unknown_response_messages.txt'
CHATBOT_LONGTERM_MEMORY = 'chatbot_longterm_memory.txt'

# Cornell corpus files
CORNELL_PATH = '{}/cornell'.format(CORPUS_DIR)
CORNELL_CONVO_FILE = 'movie_conversations.txt'
CORNELL_LINE_FILE = 'movie_lines.txt'
CORNELL_SEP = ' +++$+++ '

# Self-dialogue corpus files
SELFD_PATH = '{}/self_dialogue_corpus'.format(CORPUS_DIR)
SELFD_DATA_PATH = 'corpus'
SELFD_BLOCKED_WORKERS = 'blocked_workers.txt'
SELFD_TOPICS = ['star_wars', 'action', 'movies']

# Model specific folders for checkpoints, processed dataset, and dialogue
# history
PROCESSED_PATH = '{}/processed_dataset'.format(ROOT_DIR)
CPT_PATH = '{}/checkpoints'.format(ROOT_DIR)
OUTPUT_FILE = '{}/output_convo.txt'.format(ROOT_DIR)

# Data Pre-Processing Cleaning Config
REMOVE_CHARS = r'[#$%"\+@<=>!&,.?:;()*\[\]^_`{|}~/\t\n\r\x0b\x0c“”]'

# Encoder/Decoder reserved token IDs
PAD = '<#>'
UNK = '<?>'
SOS = '<SOS>'
EOS = '<EOS>'
PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

# Bucket sizes:
# 	(max input sequence size, max output sequence size)
# 	limit to reduce chabot complexity for both user input sequences and chatbot
# 	ouput sequences
BUCKETS = [(5, 5), (7, 7), (10, 10), (15, 15)]

# Dataset Options
TRAIN_SET_SIZE = 20000
TEST_SET_SIZE_PERCENT = 10

# Model Settings
# Optimiser Configuration - options are 'adam' (default) and 'sgd'
OPTIMISER = 'adam'
LR = 0.001
# Network Configuration
NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 128
DROPOUT = 0.9
MAX_GRAD_NORM = 5.0

# Possible search modes are 'argmax' (default) and 'beam'
# Note: Beam-search mode will take significantly longer to process a response
# than argmax due to the conversions and calculations within
DECODE_MODE = 'argmax'
BEAM_DEBUG = False
NUM_SAMPLES = 10

ENC_VOCAB = 23691
DEC_VOCAB = 23911
