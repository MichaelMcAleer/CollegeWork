# ---------------------------------------------------------------
# Natural Language Processing
# Assignment 3 - Dialogue System
# Michael McAleer R00143621
#
# >>> Improvements Made:
# - Add Self-Dialogue Corpus (github.com/jfainberg/self_dialogue_corpus)
# - Add user-defined corpus of own custom dialogue (follow designated format)
# - Add chatbot personality
# - Complete overhaul of the tokeniser
# - Changed encoder/decoder tokens to more pythonic alternatives (no invalid
#   escape sequence warning from pep-8)
# - Moved reserved tokens to config
# - LoadData now takes into consideration user set max training set size
# - Test set size set to % instead of fixed size to allow for variable dataset
#   size
# - Rename files to something better suited to not conflict with variable
#   naming conventions
# - User set training data set size
# ---------------------------------------------------------------
import csv
import numpy as np
import os
import random
import re

from glob import glob

import cb_config as cbc


# ---------------------------------------------------------------
# Step 1 - Prepare Data
# ---------------------------------------------------------------
# ---------------------------------------
# Step 1.1 - Prepare Chatbot Personality
# ---------------------------------------
def prepart_chatbot_personality():
    """Prepare the chatbot personality:
        - Specific Q/A about the chatbot
        - Movie script with chatbot responses for model decoder
        - Longterm memory (facts learned from previous user interactions)

    :return: chatbot questions & answers -- list, list
    """
    # Initialise the question and answer lists
    q, a = list(), list()
    # Initialise chatbot specific Q&A
    chatbot_qa_list = list()

    # If the chatbot specific Q&A file exists load it into a list
    cb_direct_qa_path = os.path.join(cbc.CHATBOT_PERSONALITY_DIR,
                                     cbc.CHATBOT_DIRECT_PATH)
    if os.path.isfile(cb_direct_qa_path):
        print('\t- Loading chatbot details...')
        with open(cb_direct_qa_path) as f:
            chatbot_qa_list = f.read().splitlines()
    # For each chatbot specific Q&A , add the question and answer to their
    # respective lists
    for qa in chatbot_qa_list:
        data = qa.split('|')
        q.append(data[0].strip())
        a.append(data[1].strip())

    # Define chatbot personality from source movie script excerpts, we only
    # want HALs responses to be returned from the decoder so we feed the other
    # actors dialogue into the model encoder and output Hal's response from the
    # decoder
    # If the chatbot personality dialogue file exists load it into a list
    cb_personality_path = os.path.join(cbc.CHATBOT_PERSONALITY_DIR,
                                       cbc.CHATBOT_PERSONALITY_PATH)
    if os.path.isfile(cb_personality_path):
        print('\t- Loading chatbot personality...')
        with open(cb_personality_path) as f:
            personality_dialogue = f.read().splitlines()
        # For each line in the personality dialogue file
        for i, line in enumerate(personality_dialogue):
            # Ship the line if it is a scene marker
            if line != '===':
                # Split the line
                context = line.split('|')
                # Ensure that it is not Hal that is initiating the dialogue
                if context[0] != 'Hal':
                    try:
                        # Get Hal's response from the current question
                        response = personality_dialogue[i + 1].split('|')
                        # Ensure that it is Hal responding
                        if response[0].strip() == 'Hal':
                            # Append the Q&A to their respective lists
                            q.append(context[1].strip())
                            a.append(response[1].strip())
                    except IndexError:
                        pass

    # Load chatbot longterm memory if file exists, this is a file of Q&As that
    # Hal has saved from previous user interactions as questions the user has
    # asked and corrected user answers, we want Hal to train on these 'correct'
    # facts/answers
    # Load the longterm memory file into a list if it exists
    cb_longterm_memory_path = os.path.join(cbc.CHATBOT_PERSONALITY_DIR,
                                           cbc.CHATBOT_LONGTERM_MEMORY)
    if os.path.isfile(cb_longterm_memory_path):
        print('\t- Loading chatbot longterm memory...')
        with open(cb_longterm_memory_path) as f:
            long_term_memory = f.read().splitlines()
            # For each Q&A in the longterm memory file, split it and append
            # each Q&A to their respective lists
            for i, line in enumerate(long_term_memory):
                try:
                    memory = line.split('|')
                    q.append(memory[0].strip())
                    a.append(memory[1].strip())
                except IndexError:
                    pass

    # Assert that every question for the encoder has a response from the
    # decoder
    assert len(q) == len(a)
    return q, a


# ---------------------------------------
# Step 1.2 - Prepare Cornell Dataset
# ---------------------------------------
def prepare_cornell_data():
    """Prepare the Cornell dataset.

    :return: questions (encoder), answers (decoder) -- tuple(list, list)
    """
    # Get the IDs in each line
    id2line = get_lines()
    # Get the conversations from the dataset
    convos = get_convos()
    # Return lists of Q&As by converting line IDs to conersation dialogues
    return question_answers(id2line, convos)


def get_lines():
    """Get the line IDs from the Cornell dataset and the respective lines of
    dialogue.

    :return: line id, dialouge -- dict
    """
    id2line = dict()
    file_path = os.path.join(cbc.CORNELL_PATH, cbc.CORNELL_LINE_FILE)
    with open(file_path, 'r', errors='ignore') as f:
        i = 0
        try:
            for line in f:
                parts = line.split(cbc.CORNELL_SEP)
                if len(parts) == 5:
                    if parts[4][-1] == '\n':
                        parts[4] = parts[4][:-1]
                    id2line[parts[0]] = parts[4]
                i += 1
        except UnicodeDecodeError:
            print(i, line)
    return id2line


def get_convos():
    """Get conversations by line id from the Cornell dataset.

    :return: dialogues -- nested list
    """
    file_path = os.path.join(cbc.CORNELL_PATH, cbc.CORNELL_CONVO_FILE)
    convos = list()
    with open(file_path, 'r') as f:
        for l1 in f.readlines():
            parts = l1.split(cbc.CORNELL_SEP)
            if len(parts) == 4:
                convo = []
                for l2 in parts[3][1:-2].split(', '):
                    convo.append(l2[1:-1])
                convos.append(convo)

    return convos


def question_answers(id2line, convos):
    """Construct dialogues from line ids and seperate into questions and
    answers.

    :param id2line: id to dialogue mapping -- dict
    :param convos: dialogue ids -- nested list
    :return: dialogue questions & answers -- list, list
    """
    q, a = list(), list()
    for convo in convos:
        for index, line in enumerate(convo[:-1]):
            q.append(id2line[convo[index]])
            a.append(id2line[convo[index + 1]])
    assert len(q) == len(a)
    return q, a


# ---------------------------------------
# Step 1.3 - Prepare Self Dialogue Dataset
# ---------------------------------------
def prepare_self_dialogue_data():
    """Prepare the self-dialogue dataet:
        - Get the list of blocked workers to exclude from dataset
        - Define the dataset topics to load
        - Read the dataset
        - Extract Q&A pairs

    :return: dialogue questions & answers -- list, list
    """
    # Initialise the question and answer lists
    q, a = list(), list()

    # Get the blocked workers list
    blocked_workers_path = os.path.join(cbc.SELFD_PATH,
                                        cbc.SELFD_BLOCKED_WORKERS)
    blocked_workers = set()
    if os.path.isfile(blocked_workers_path):
        print('\t- Loading blocked workers file...')
        with open(blocked_workers_path) as f:
            blocked_workers = set(f.read().splitlines())

    # Define Topics
    # Set the path to the self-deialogue corpus directory
    corpus_data_path = os.path.join(cbc.SELFD_PATH, cbc.SELFD_DATA_PATH)
    # Extract the list of topics by reading directory names in the corpus
    # parent directory
    corpus_topics = os.listdir(corpus_data_path)
    # Initialise selected topics list
    selected_topics = list()
    # If the user has specified topics for loading
    if cbc.SELFD_TOPICS:
        print('\t- Loading user-defined self-dialogue topics...')
        # For each topic specified, if it exists in the corpus add it to the
        # list of extracted topics
        for topic in cbc.SELFD_TOPICS:
            if topic in corpus_topics:
                selected_topics.append(topic)
    # Else no user specified topics, load all topics
    else:
        selected_topics = corpus_topics

    # Read corpus data
    # Initialise global dialogue dict
    conversation_data = dict()
    # For each selected topic
    for topic in selected_topics:
        print('\t- Loading topic: {}'.format(topic))
        # Get the path to each topic directory
        process_path = '{c}/{t}'.format(c=corpus_data_path, t=topic)
        # Initialise the topic dict
        dialogues = dict()
        # For each file in the topic directory
        for filename in glob("{0}/*.csv".format(process_path)):
            # Open the file and read it into a CSV dict
            with open(filename, encoding="utf8") as corpus_file:
                csv_file = csv.DictReader(corpus_file)
                # For each row in the file
                for row in csv_file:
                    # If the row is not marked as rejected or is a blocked
                    # worker, add the conversation dialogue and unique ID to
                    # topic dialogue
                    if not row['Reject'] and (
                            row['WorkerId'] not in blocked_workers):
                        dialogues[row['AssignmentId']] = row
        # Update the global conversation dict with the topic conversation data
        conversation_data.update(dialogues)

    # Extract Question/Answer Pairs
    # Get a list of unique conversation IDs
    conversation_key_list = conversation_data.keys()
    # For each conversation ID
    for conversation_key in conversation_key_list:
        # Get the associated conversation
        dialougue = conversation_data.get(conversation_key)
        # Get the length of the dialogue
        dialogue_len = len(
            [key for key in dialougue.keys() if key.startswith('Answer.')])
        # For each Q&A
        for i in range(1, dialogue_len):
            # Append the Q&A to their respective lists
            q.append(dialougue.get('Answer.sentence{x}'.format(x=i)))
            a.append(dialougue.get('Answer.sentence{x}'.format(x=i + 1)))

    # Assert that every question for the encoder has a response from the
    # decoder
    assert len(q) == len(a)
    return q, a


# ---------------------------------------
# Step 1.4 - Prepare Dataset
# ---------------------------------------
def prepare_dataset(question_list, answer_list, p_reserve):
    """Take each Q&A from the loaded corpora, extract a percentage of data to
    use as training data, clean the data and write to respective train/test
    encoder and decoder files for further processing later.

    :param question_list: all loaded questions -- list
    :param answer_list: all loaded answers -- list
    :param p_reserve: personality reserve count -- int
    """
    # Create directory to store processed dataset
    make_dir(cbc.PROCESSED_PATH)

    # Set the training and test dataset sizes
    train_size = cbc.TRAIN_SET_SIZE if cbc.TRAIN_SET_SIZE > 0 else (
        len(question_list))
    test_set_size = (int(train_size / 100) * cbc.TEST_SET_SIZE_PERCENT)
    # Get random sample of test indexes to extract coversations from dataset,
    # p_reserve will ensure the chatbot personality data is not included in
    # the test data, we want the model to train on this only
    test_ids = random.sample(
        [i for i in range(p_reserve + 1, len(question_list))], test_set_size)

    # Set the train/test encoder and decoder filenames
    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    # Initialise open file list
    files = []
    # Open each encoder and decoder file for writing
    for filename in filenames:
        files.append(
            open(os.path.join(cbc.PROCESSED_PATH, filename), 'w'))
    # For each question in our complete loaded corpora
    for i in range(len(question_list)):
        # Clean the question and answer
        question = clean_text(question_list[i])
        answer = clean_text(answer_list[i])
        # If the question index is in the list of test indexes, write to
        # test encoder and decoder files
        if i in test_ids:
            files[2].write(question + '\n')
            files[3].write(answer + '\n')
        # Else write to train encoder and decoder files
        else:
            files[0].write(question + '\n')
            files[1].write(answer + '\n')
    # Close each file
    for file in files:
        file.close()


# ---------------------------------------
# Clean dataset by line
# ---------------------------------------
def clean_text(line_in):
    """Clean a line of data from the dataset.

    :param line_in: dialogue line -- str
    :return: cleaned dialogue line -- str
    """

    def _is_english(s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    # Remove any new-line characters
    line = line_in.replace('\n', '')
    # Clean any erroneous markers from Cornell dataset
    line = line.replace('<u>', '')
    line = line.replace('</u>', '')
    # Split line into respective tokens and remove any unwanted characters
    all_chars = [re.sub(cbc.REMOVE_CHARS, '', t) for t in line.split()]
    # Filter empty strings from tokens
    filter_tokens = list(filter(None, all_chars))
    # Remove non-english words
    english_words = [c for c in filter_tokens if _is_english(c)]
    # Remove any words with that have anything other than alpha/numeric chars,
    # hyphens, or apostrophes
    alpha_words = [c for c in english_words if re.match(r"[-'a-zA-Z1-9]", c)]
    # Normalise text by changing it all to lower case
    normal_text = [word.lower() for word in alpha_words]
    # Rejoin tokens into string
    return ' '.join(normal_text)


def make_dir(path):
    """Create a directory if it doesn't exist already.

    :param path: path to directory -- str
    """
    try:
        os.mkdir(path)
    except OSError:
        pass


# ---------------------------------------------------------------
# Step 2 - Process Data
# ---------------------------------------------------------------
def process_data():
    """Process the prepared dataset.
        - Build dataset vocabularys
        - Convert tokens to Ids for each file
    """
    print('Preparing data to be model-ready ...')
    # Build vocabulary for both train encoder and decoder files
    build_vocab('train.enc')
    build_vocab('train.dec')
    # Convert train/test encoder and decoder files to id equivalent files
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')


# ---------------------------------------
# Step 2.1 - Build Vocab
# ---------------------------------------
def build_vocab(filename):
    """Build a vocabulary for a given question or answer file, write the vocab
    to file once done.

    :param filename: filename to process -- str
    """
    # Set the path for file load
    in_path = os.path.join(cbc.PROCESSED_PATH, filename)
    # Set the path for file save
    out_path = os.path.join(cbc.PROCESSED_PATH,
                            'vocab.{}'.format(filename[-3:]))
    # Initialise vocab dict
    vocab = dict()
    # Open file
    with open(in_path, 'r') as f:
        # Read file contents into a list, for each line in the file
        for line in f.readlines():
            # Tokenise the row to get list of tokens
            for token in basic_tokenizer(line):
                # Remove any new-line characters
                token = token.replace('\n', '')
                # If the token does not yet exist in the vocab, add it to the
                # dict with a counter of 0
                if token not in vocab:
                    vocab[token] = 0
                # Else word exists, increment counter
                vocab[token] += 1

    # Sort the vocab so most frequently occuring words are at the top of the
    # file to reduce processing/fetch time later
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    # Remove any possible lingering duplicates by casting list to set then
    # back again
    set_vocab = list(set(sorted_vocab))
    # Open the output file for writing
    with open(out_path, 'w') as f:
        # Add the reserve tokens for padding, unknown words, start and end of
        # sentence markers
        f.write(cbc.PAD + '\n')
        f.write(cbc.UNK + '\n')
        f.write(cbc.SOS + '\n')
        f.write(cbc.EOS + '\n')
        # Set vocab size from here
        count = 4
        # For each word in the vocab dict, write it to file and increment
        # vocab count by one
        for word in set_vocab:
            f.write(word + '\n')
            count += 1
        # Write the size of the encoder and decoder vocabs to the config file
        with open('cb_config.py', 'a') as cf:
            if filename[-3:] == 'enc':
                cf.write('ENC_VOCAB = ' + str(count) + '\n')
            else:
                cf.write('DEC_VOCAB = ' + str(count) + '\n')


def basic_tokenizer(line):
    """Split a line into tokens using basic line split on whitespace, all
    data has been cleaned before this point in clean_text() so complex
    tokeniser is not required (NLTK and PyContractions were implemented but the
    impact on data processing time did not make inclusion worthwhile).

    :param line: line from dataset -- str
    :return: line tokens -- list
    """
    return line.split(' ')


# ---------------------------------------
# Step 2.2 - Token 2 ID
# ---------------------------------------
def token2id(data, mode):
    """Convert all dialogue into ids from respective vocab in dataset.

    :param data: dataset in use (train/test) -- str
    :param mode: dataset type in use (encoder/decoder) -- str
    """
    # Set vocab filename, dataset input path and dialogue id file output path
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    # Load the vocabulary into a dict
    _, vocab = load_vocab(
        os.path.join(cbc.PROCESSED_PATH, vocab_path))
    # Open the vocab file
    in_file = open(os.path.join(cbc.PROCESSED_PATH, in_path), 'r')
    # Open the dialogue id file
    out_file = open(os.path.join(cbc.PROCESSED_PATH, out_path), 'w')

    # Read each line in the dataset into a lis
    lines = in_file.read().splitlines()
    # For each line in the dataset
    for line in lines:
        # If in decoder mode
        if mode == 'dec':
            # Start the coding with the SOS marker
            ids = [vocab[cbc.SOS]]
        # Else in encoder mode, start with empty list
        else:
            ids = []
        # Extend the list of ids with each token id from the line of dialogue
        ids.extend(sentence2id(vocab, line))
        # If in decoder mode end the list of ids with the EOS marker
        if mode == 'dec':
            ids.append(vocab[cbc.EOS])
        # Write the line of token ids to file
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')


def load_vocab(vocab_path):
    """Load vocab file.

    :param vocab_path: path to vocab file -- str
    :return: words -- list, dict
    """
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    """Convert a dialogue sentence to respective token ids.

    :param vocab: vocab -- dict
    :param line: dialogue sentence -- str
    :return: sentence token ids -- list
    """
    return (
        [vocab.get(token, vocab[cbc.UNK]) for token in basic_tokenizer(line)])


# ---------------------------------------------------------------
# Step 2 - Chatbot Assistive Functions
# ---------------------------------------------------------------
def load_data(enc_filename, dec_filename):
    """Load the dataset into sequence length designated buckets.

    :param enc_filename: encoder filename -- str
    :param dec_filename: decoder filename -- str
    :return: data buckets -- nested list
    """
    # Open the encoder and decoder id files for reading
    encode_file = open(
        os.path.join(cbc.PROCESSED_PATH, enc_filename), 'r')
    decode_file = open(
        os.path.join(cbc.PROCESSED_PATH, dec_filename), 'r')
    # Read the encoder and decoder files into lists
    encode, decode = encode_file.readlines(), decode_file.readlines()
    # Initialise nested list to hold data in buckets
    data_buckets = [list() for _ in cbc.BUCKETS]

    # Set the training dataset size
    train_size = cbc.TRAIN_SET_SIZE if cbc.TRAIN_SET_SIZE > 0 else (
        len(encode_file.readlines()))
    # Initialise the index and counter
    i, cnt = 0, 0
    # For each line of data in the encoder (and decoder of same length)
    for i in range(0, len(encode)):
        # If the training size limit has not exceeded the training dataset size
        # limit set by the user in the config
        if cnt <= train_size:
            # Output progress of data bucketing
            if (i + 1) % 10000 == 0:
                print("Bucketing conversation number", i + 1)
            # Get the list of ids from the encoder and decoder dialogue
            # sentences
            encode_ids = [int(id_) for id_ in encode[i].split()]
            decode_ids = [int(id_) for id_ in decode[i].split()]
            # Get the bucket id and encoder/decoder max sequence lengths
            for bucket_id, (encode_max_size, decode_max_size) in (
                    enumerate(cbc.BUCKETS)):
                # If the length of the encoder/decoder sequence doesn't exceed
                # the bucket max length, add the dialogue sentences to the data
                # buckets, else skip to the next bucket and repeat
                if len(encode_ids) <= encode_max_size and (
                        len(decode_ids) <= decode_max_size):
                    data_buckets[bucket_id].append([encode_ids, decode_ids])
                    cnt += 1
                    break
            i += 1

    return data_buckets


def get_batch(data_bucket, bucket_id, chat_mode=False):
    """Return one batch of data to feed into the model in the required input
    format for the LSTM model.

    :param data_bucket: dataset data bucked -- nested list
    :param bucket_id: bucket id -- int
    :param chat_mode: if chat mode is enabled -- bool
    :return: encoder inputs, decoder inputs, masks -- list, list, list
    """

    def _pad_input(input_, size_):
        """Pad any short input to standardise sequence length thoughout model
        inputs."""
        return input_ + [cbc.PAD_ID] * (size_ - len(input_))

    def _reshape_batch(input_, size_, bsize_):
        """Reshape batch to invert axis."""
        batch_inputs = []
        for x in range(size_):
            batch_inputs.append(np.array([input_[y][x] for y in range(bsize_)],
                                         dtype=np.int32))
        return batch_inputs

    # Get the training batch size
    train_batch = cbc.BATCH_SIZE if cbc.BATCH_SIZE else 128
    batch_size = 1 if chat_mode else train_batch
    # Get the max length of the encoder input and decoder output for the bucket
    encoder_size, decoder_size = cbc.BUCKETS[bucket_id]
    # Initialise encoder and decoder input lists
    encoder_inputs, decoder_inputs = list(), list()

    # For input in range of the batch size
    for _ in range(batch_size):
        # Get a random encoder and decoder sentence from the databucket
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(
            list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # Invert the shape of the encoder and decoder inputs
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size,
                                          batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size,
                                          batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = list()
    # For each id in the decoder output length
    for length_id in range(decoder_size):
        # Create a np array of zeros equal to decoder length
        batch_mask = np.ones(batch_size, dtype=np.float32)
        # For length of batch
        for batch_id in range(batch_size):
            # Set the target
            target = None
            # The corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            # Set mask to 0 if the corresponding target is a PAD symbol.
            if length_id == decoder_size - 1 or target == cbc.PAD_ID:
                batch_mask[batch_id] = 0.0
        # Add the mask to the list of masks
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks


def main():
    """Step 1 - Prepare Data."""
    # 1.1 - Prepare Chatbot Personality
    print('Reading Chatbot Personality Data...')
    cb_q, cb_a = prepart_chatbot_personality()
    # 1.2 - Prepare Cornell Data
    print('Reading Cornell Dataset...')
    co_q, co_a = prepare_cornell_data()
    # 1.3 - Prepart Self-Dialogue Dataset
    print('Reading Self-Dialogue Dataset...')
    sd_q, sd_a = prepare_self_dialogue_data()
    # Merge datasets into combined singular datset
    questions = cb_q + co_q + sd_q
    answers = cb_a + co_a + sd_a
    # Prepare Training & Test Encoder/Decoder
    print('Preparing Dataset...')
    prepare_dataset(questions, answers, len(cb_q))
    """Step 2 - Process Data."""
    print('Processing Dataset...')
    process_data()
    print('Dataset Ready')


if __name__ == '__main__':
    main()
