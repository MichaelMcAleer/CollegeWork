# ---------------------------------------------------------------
# Natural Language Processing
# Assignment 3 - Dialogue System
# Michael McAleer R00143621
#
# >>> Improvements Made:
# - Add personality exit messages
# - Add long/short term memory for corrections
# - Add ability to recall important details about user
# - Add custom output message writer
# ---------------------------------------------------------------
import argparse
import math
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time

import cb_config
import cb_data
from cb_model import ChatBotModel
from cb_session import ChatSession

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


# ---------------------------------------------------------------
# Train Model
# ---------------------------------------------------------------
def train():
    """Train the chatbot."""
    # Load test and training buckets (datasets by sequence length)
    test_buckets, train_buckets, train_buckets_scale = _get_buckets()
    # Initialise ChatBot model, set forward_only to False because backward pass
    # is required in training mdoe
    model = ChatBotModel(False, cb_config.BATCH_SIZE)
    # Build model loss, optimiser, summary etc.
    model.build_graph()
    # Initialise TF saver
    saver = tf.train.Saver()
    # Initialise TF session
    with tf.Session() as sess:
        print('Running session')
        # Initialise TF global variables
        sess.run(tf.global_variables_initializer())
        # Restore model parameters from previous run if they exist
        _check_restore_parameters(sess, saver)
        # Get training iteration step number
        iteration = model.global_step.eval()
        # Initialise total model loss
        total_loss = 0
        while True:
            # Set how long the model should train for before saving
            save_step = _get_save_step(iteration)
            # Get a random bucket ID
            bucket_id = _get_random_bucket(train_buckets_scale)
            # Get model inputs (encoder inputs, decoder inputs, masks)
            encoder_inputs, decoder_inputs, decoder_masks = cb_data.get_batch(
                train_buckets[bucket_id], bucket_id)
            # Start timer
            start = time.time()
            # Run a single training step
            _, step_loss, _ = run_step(
                sess, model, encoder_inputs, decoder_inputs, decoder_masks,
                bucket_id, False)
            # Add loss to total loss counter
            total_loss += step_loss
            # Increment iteration counter
            iteration += 1
            # If the iteration count equals the save point
            if iteration % save_step == 0:
                # Output model concise summary of progress so far
                print('Iter {i}: loss {l}, time {t}'.format(
                    i=iteration, l=total_loss / save_step,
                    t=time.time() - start))
                # Reset the total loss counter
                total_loss = 0
                # Save the model in the
                saver.save(sess, os.path.join(cb_config.CPT_PATH, 'chatbot'),
                           global_step=model.global_step)
                # Evaluate the model on the test data if count condition is met
                if iteration % (10 * save_step) == 0:
                    # Run evals on development set and print their loss
                    _eval_test_set(sess, model, test_buckets)
                # Flush output buffer
                sys.stdout.flush()


# ---------------------------------------
# Training Mode - Get Dataset Buckets
# ---------------------------------------
def _get_buckets():
    """Load the dataset into buckets based on their lengths.

    :return: test buckets, train buckets, training scale -- list, list, list
    """
    # Load the test dataset
    test_buckets = cb_data.load_data('test_ids.enc', 'test_ids.dec')
    # Load the training dataset
    train_buckets = cb_data.load_data('train_ids.enc', 'train_ids.dec')
    # Get the training bucket sizes (max input and output sequence size)
    train_bucket_sizes = [len(train_buckets[b]) for b in
                          range(len(cb_config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    # Ge the max train bucket size
    train_total_size = sum(train_bucket_sizes)
    # List of increasing numbers from 0 to 1 that we'll use to select a bucket
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, train_buckets, train_buckets_scale


# ---------------------------------------
# Training Mode - Restore Previous Model Parameters
# ---------------------------------------
def _check_restore_parameters(sess, saver):
    """Restore the previously trained parameters if they exist.

    :param sess: tensorflow session -- session object
    :param saver: tensorflow saver -- saver object
    """
    # Load checkpoint state if it exists in checkpoint directory
    ckpt = tf.train.get_checkpoint_state(
        os.path.dirname(cb_config.CPT_PATH + '/checkpoint'))
    # If the checkpoint exists restore the model parameters
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    # Else model is initialised with fresh parameters
    else:
        print("Initializing fresh parameters for the Chatbot")


# ---------------------------------------
# Training Mode - Training Operations
# ---------------------------------------
def _get_save_step(iteration):
    """How many steps should the model train before it saves a model
    checkpoint.

    :param iteration: iteration count -- int
    :return: iteration save point -- int
    """
    if iteration < 100:
        return 30
    return 100


def _get_random_bucket(train_buckets_scale):
    """Get a random bucket from which to choose a training sample.

    :param train_buckets_scale: bucket scales -- list
    :return: bucket id -- int
    """
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])


def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs,
                    decoder_masks):
    """Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths.

    :param encoder_size: encoder size -- int
    :param decoder_size: decoder size --int
    :param encoder_inputs: encoder inputs -- list
    :param decoder_inputs: decoder inputs -- list
    :param decoder_masks: decoder masks -- list
    :raises: ValueError
    """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_masks), decoder_size))


# ---------------------------------------
# Training Mode - Run Single Train Step
# ---------------------------------------
def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks,
             bucket_id, forward_only):
    """Run one step in training.

    :param sess: tensorflow session -- session object
    :param model: chatbot model -- model object
    :param encoder_inputs: encoder inputs -- list
    :param decoder_inputs: decoder inputs -- list
    :param decoder_masks: decoder masks -- list
    :param bucket_id: bucket id -- int
    :param forward_only: forward pass only for chat or evaluate mode -- bool
    :return: output feed -- nested list
    """
    # Get the encoder and decoder max size from cb_config bucket
    encoder_size, decoder_size = cb_config.BUCKETS[bucket_id]
    # Assert various lengths are equal to avoid issues later in with
    # incorrect data decoding
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs,
                    decoder_masks)
    # Initialise model input feed
    input_feed = dict()
    # Add encoder, decoder and mask data to input feed
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]
    # Set input feed last target (not sure about the purpose of this - check
    # further
    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # Set output feed - dependent on wheter or not backward pass is enables
    if not forward_only:
        # Output optimiser, gradient norm, loss
        output_feed = [model.train_ops[bucket_id],
                       model.gradient_norms[bucket_id],
                       model.losses[bucket_id]]
    else:
        # Output loss and output logits
        output_feed = [model.losses[bucket_id]]
        for step in range(decoder_size):
            output_feed.append(model.outputs[bucket_id][step])

    # Run one step of the model
    outputs = sess.run(output_feed, input_feed)
    # If training mode and backward pass enabled return gradient norm and loss
    if not forward_only:
        return outputs[1], outputs[2], None
    # If chat mode and forward pass only output loss and output logits
    else:
        return None, outputs[0], outputs[1:]


# ---------------------------------------
# Training Mode - Evaluate Model on test set
# ---------------------------------------
def _eval_test_set(sess, model, test_buckets):
    """Evaluate on the test set.

    :param sess: tensorflow session -- session object
    :param model: chatbot model -- model object
    :param test_buckets: test dataset buckets -- nested list
    """
    # For each dataset bucket
    for bucket_id in range(len(cb_config.BUCKETS)):
        # If the bucket is empty skip it and continue to next bucket
        if len(test_buckets[bucket_id]) == 0:
            print("  Test: empty bucket %d" % bucket_id)
            continue
        # Start eval timer
        start = time.time()
        # Get the model inputs
        encoder_inputs, decoder_inputs, decoder_masks = cb_data.get_batch(
            test_buckets[bucket_id], bucket_id)
        # Get the loss value from the current eval step and output
        _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs,
                                   decoder_masks, bucket_id, True)
        print('Test bucket {b}: loss {l}, time {t}'.format(
            b=bucket_id, l=step_loss, t=time.time() - start))


# ---------------------------------------------------------------
# Chat with model
# ---------------------------------------------------------------
def chat():
    """Chat with the Chat Bot (model forward pass only)."""
    # Load the encoder vocab
    _, enc_vocab = cb_data.load_vocab(
        os.path.join(cb_config.PROCESSED_PATH, 'vocab.enc'))
    # Load the decoder vocab
    inv_dec_vocab, _ = cb_data.load_vocab(
        os.path.join(cb_config.PROCESSED_PATH, 'vocab.dec'))
    # Initialise ChatBot with forward-pass only set to True
    model = ChatBotModel(True, batch_size=1)
    # Build model loss, optimiser, summary etc.
    model.build_graph()
    # Initialise TF saver
    saver = tf.train.Saver()
    # Initialise Chat Session
    chat_session = ChatSession()
    # Initialise TF session
    with tf.Session() as sess:
        # Initialise TF global variables
        sess.run(tf.global_variables_initializer())
        # Restore model parameters from previous run
        _check_restore_parameters(sess, saver)
        # Open the output file for chatbot dialogue history
        output_file = open(
            os.path.join(
                cb_config.PROCESSED_PATH, cb_config.OUTPUT_FILE), 'a+')
        # Get the max input sequence length
        max_length = cb_config.BUCKETS[-1][0]
        # Output welcome messages
        print(" HAL: Good evening, I am the H.A.L 9000. You may call me Hal.\n"
              " HAL: Press Enter to exit, max input length is {m}\n"
              "=========================================".format(m=max_length))
        # Run the user-input/response loop until user exits
        while True:
            # Get the input from the user
            line = _get_user_input()
            chat_session.remember_short_term(' HUMAN: ' + line)

            # If the input has content and finishes with newline
            if len(line) > 0 and line[-1] == '\n':
                # Get the input data minus the newline
                line = line[:-1]
                # Write the human input to the history file
                output_file.write(' HUMAN: ' + line + '\n')
            # If the input is empty (Enter key-press) exit with message
            if line == '':
                _print_exit_message(chat_session, output_file)
                break
            # Determine if user inputs info that should be saved to memory
            if _process_for_info(chat_session, output_file, line):
                continue
            # Reply from memory if user asks relevant question
            if _reply_to_session_question(chat_session, output_file, line):
                continue

            # Get token-ids for the input sentence.
            token_ids = cb_data.sentence2id(enc_vocab, str(line))
            # If the user input is larger than the max sequence length skip the
            # input and prompt for fresh input
            if len(token_ids) > max_length:
                msg = (" HAL: Im afraid I can't do that {n}, max length I can "
                       "handle is: {m}".format(n=chat_session.name,
                                               m=max_length))
                _output_message(chat_session, msg)
                output_file.write(' BOT: {m}\n'.format(m=msg))
                continue
            # Retrieve the correct data bucket by input sequence length
            bucket_id = _find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = cb_data.get_batch(
                [(token_ids, [])], bucket_id, chat_mode=True)
            # Get output logits for the sentence.
            _, _, output_logits = run_step(
                sess, model, encoder_inputs, decoder_inputs,
                decoder_masks, bucket_id, True)
            # Construct the ChatBot repsonse to the user-input
            response = _construct_response(output_logits, inv_dec_vocab)
            # Hal wasn't able to construct a response or constructed a no
            # word response, override and output an unknown response message
            if response in ['', ' ']:
                _print_no_found_response_message(chat_session, output_file)
                continue
            _output_message(chat_session, 'HAL: {r}'.format(r=response))
            # Write the bot output to the history log
            output_file.write(' BOT: ' + response + '\n')
        # When the dialogue loop finishes mark the log with a seperator and
        # close the file
        output_file.write('=========================================\n')
        output_file.close()
        # Write long-term memory to file
        if chat_session.long_term_memory:
            print('=========================================')
            chat_session.write_long_term_memory_to_file()


# ---------------------------------------
# Chat Mode - Get user input
# ---------------------------------------
def _get_user_input():
    """Get user input for chatbot dialogue.

    :return: user input -- str
    """
    print(" > ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


# ---------------------------------------
# Chat Mode - Process user input for saveable information
# ---------------------------------------
def _process_for_info(_session, _outfile, _input):
    """Process user input for information that should be saved to short or long
    term memory.

    :param _session: user chatbot session -- chatbot session object
    :param _input: user input -- str
    :return: if data was extracted from the user input -- bool
    """
    # Clean the user input
    clean_input = cb_data.clean_text(_input)
    # Tokenise the user input
    tokens = cb_data.basic_tokenizer(clean_input)

    # If the user tells the chatbot their name...
    if all(x in tokens for x in ['my', 'name', 'is']) and 'what' not in tokens:
        # We assume the name will follow 'is' in the list of tokens, if the
        # user tells us their full name we remember all tokens following 'is'
        name = ' '.join(tokens[tokens.index('is') + 1:])
        _session.name = name.capitalize()

        msg = 'Hello {n}'.format(n=_session.name)
        _output_message(_session, 'Hal: {m}'.format(m=msg))
        _outfile.write(' BOT: {m}\n'.format(m=msg))
        return True

    # If the user tells us their favourite thing
    if all(x in tokens for x in ['my', 'favourite']) and 'what' not in tokens:
        # We assume the favourite 'thing' category follows 'favourite' in the
        # list of tokens
        i_cat = tokens.index('favourite')
        cat = tokens[i_cat + 1]
        # We assume the favourite 'thing' follows is, we take all following
        # tokens incase it spans more than one word (ex: film -> star wars)
        thing = ' '.join(tokens[tokens.index('is') + 1:])
        # Save the favourite thing in the chat session favourites dict by
        # category
        _session.favourites[cat] = thing.capitalize()

        msg = 'Interesting...'
        _output_message(_session, 'Hal: {m}'.format(m=msg))
        _outfile.write(' BOT: {m}\n'.format(m=msg))
        return True

    # If the user tells us how old they are
    if all(x in tokens for x in ['years', 'old']) and 'how' not in tokens:
        # We assume the age will follow 'am' in the list of tokens
        age = tokens[tokens.index('am') + 1]
        _session.age = age

        msg = 'I am 52 years old'
        _output_message(_session, 'Hal: {m}'.format(m=msg))
        _outfile.write(' BOT: {m}\n'.format(m=msg))
        return True

    # If the user tells us where they are from
    if all(x in tokens for x in ['am', 'from']) and 'where' not in tokens:
        # We assume that their home location follows 'from' in the token list
        location = ' '.join(tokens[tokens.index('from') + 1:])
        _session.location = location.capitalize()
        msg = 'I am from the United States Spacecraft Discovery One'
        _output_message(_session, 'Hal: {m}'.format(m=msg))
        _outfile.write(' BOT: {m}\n'.format(m=msg))
        return True

    # If the user teaches the chatbot what the correct answer is to an
    # incorrect chatbot response
    catch = ['incorrect', 'the', 'correct', 'answer', 'is']
    if all(x in tokens for x in catch):
        # Is will appear more than once if user inputs 'that is' instead of
        # 'thats' so we need to assume that the answer will follow the last
        # 'is' in the sequence of user input tokens
        indices = [i for i, x in enumerate(tokens) if x == "is"]
        answer = ' '.join(tokens[max(indices) + 1:])
        # If an answer can be extracted
        if answer:
            # Get the question that the chatbot answered incorrectly from
            # short term memory
            question = _session.get_previous_user_question()
            # Remove the the sentence prefix
            question = question.replace('HUMAN: ', '')
            # Remove any new-line character
            question = question.replace('\n', '')
            # Add the question and correct answer to the chatbot session long
            # term memory
            _session.remember_long_term(question, answer)
            # Output message to user after question answer has been commited
            # to memory
            msg = ("Thanks for helping me learn '{l}', how can I help now "
                   "{n}?".format(l=question, n=_session.name))
            _output_message(_session, 'Hal: {m}'.format(m=msg))
            _outfile.write(' BOT: {m}\n'.format(m=msg))
        return True

    return False


# ---------------------------------------
# Chat Mode - Reply to user if they ask question to test memory
# ---------------------------------------
def _reply_to_session_question(_session, _outfile, _input):
    """Reply to a user from memory is they ask a question that will test the
    chatbot memory function.

    :param _session: user chatbot session -- chatbot session object
    :param _input: user input -- str
    :return: if data was extracted from the user input -- bool
    """
    # Clean the user input
    clean_input = cb_data.clean_text(_input)
    # Tokenise the user input
    tokens = cb_data.basic_tokenizer(clean_input)

    # If the user prompts the chatbot to answer questions about the user
    # from memory
    if all(x in tokens for x in ['what', 'is', 'my']):
        # If the user asks what their name is
        if 'name' == tokens[-1]:
            if _session.name != 'Dave':
                msg = 'Your name is {n}'.format(n=_session.name)
            else:
                msg = ("I am programmed to call you Dave, but you could "
                       "confirm your name for me with 'my name is ___'")
            _output_message(_session, 'Hal: {m}'.format(m=msg))
            _outfile.write(' BOT: {m}\n'.format(m=msg))
            return True

        # If the user asks what their age is
        if 'age' == tokens[-1]:
            if _session.age:
                msg = 'You are {a} years old {n}'.format(
                    a=_session.age, n=_session.name)
            else:
                msg = "I don't know your age yet"
            _output_message(_session, 'Hal: {m}'.format(m=msg))
            _outfile.write(' BOT: {m}\n'.format(m=msg))
            return True

        # If the user asks what their favourite 'thing' is
        if 'favourite' == tokens[-2]:
            if _session.favourites.get(tokens[-1]):
                msg = 'Your favourite {f} is {t}'.format(
                    f=tokens[-1],
                    t=_session.favourites[tokens[-1]])
            else:
                msg = "I don't know what your favourite {f} is yet".format(
                    f=tokens[-1])
            _output_message(_session, 'Hal: {m}'.format(m=msg))
            _outfile.write(' BOT: {m}\n'.format(m=msg))
            return True

    # If the user asks what their age is alternative
    if all(x in tokens for x in ['what', 'age', 'am', 'i']):
        if _session.age:
            msg = 'You are {a} years old'.format(a=_session.age)
        else:
            msg = "I don't know your age yet"
        _output_message(_session, 'Hal: {m}'.format(m=msg))
        _outfile.write(' BOT: {m}\n'.format(m=msg))
        return True

    # If the user asks what their age is alternative
    if all(x in tokens for x in ['how', 'old', 'am', 'i']):
        if _session.age:
            msg = 'You are {a} years old'.format(a=_session.age)
        else:
            msg = "I don't know your age yet"
        _output_message(_session, 'Hal: {m}'.format(m=msg))
        _outfile.write(' BOT: {m}\n'.format(m=msg))
        return True

    # If the user asks where they are from
    if all(x in tokens for x in ['where', 'am', 'i', 'from']):
        if _session.location:
            msg = 'You are from {f}'.format(f=_session.location)
        else:
            msg = "I don't know where you are from yet"
        _output_message(_session, 'Hal: {m}'.format(m=msg))
        _outfile.write(' BOT: {m}\n'.format(m=msg))
        return True

    return False


# ---------------------------------------
# Chat Mode - Exit message
# ---------------------------------------
def _print_exit_message(_session, _outfile):
    """Print chatbot themed exit message when user quits the session.

    :param _session: user chatbot session -- chatbot session object
    """
    # Get the path to the exit message file
    exit_msg_path = os.path.join(cb_config.CHATBOT_PERSONALITY_DIR,
                                 cb_config.CHATBOT_EXIT_PATH)
    # If the file exists open it
    if os.path.isfile(exit_msg_path):
        with open(exit_msg_path) as f:
            # Read the file contents into a list
            msgs = f.readlines()
            # Select a random integer and output an exit message to the user
            select = random.randint(0, len(msgs) - 1)
            # Replace 'Dave' with the current users name if they have told the
            # chat bot what it is
            msg = msgs[select].replace('Dave', _session.name)
            _output_message(_session, 'Hal: {m}'.format(m=msg))
            _outfile.write(' BOT: {m}\n'.format(m=msg))
    # Else no exit message file, print default message
    else:
        _output_message(_session, 'HAL: Good-bye {n}!'.format(n=_session.name))
        _outfile.write(' BOT: Good-bye {n}\n'.format(n=_session.name))


def _print_no_found_response_message(_session, _outfile):
    """Print chat bot themed unknown message when a response can't be predicted
    or an empty response is predicted.
    """
    # Get the path to the unknown message file
    unkown_msg_path = os.path.join(cb_config.CHATBOT_PERSONALITY_DIR,
                                   cb_config.CHATBOT_UNKNOWN_RESPONSE_PATH)
    # If the file exists open it
    if os.path.isfile(unkown_msg_path):
        with open(unkown_msg_path) as f:
            # Read the file contents into a list
            msgs = f.readlines()
            # Select a random integer and output an unknown message to the user
            select = random.randint(0, len(msgs))
            # Replace 'Dave' with the current users name if they have told the
            # chat bot what it is
            msg = msgs[select].replace('Dave', _session.name)
            _output_message(_session, 'HAL: {m}'.format(m=msg))
            _outfile.write(' BOT: {m}\n'.format(m=msg))
    # Else no unknown message file, print default message
    else:
        msg = "I'm afraid I don't understand {n}...".format(
            n=_session.name)
        _output_message(_session, 'HAL: {m}'.format(m=msg))
        _outfile.write(' BOT: {m}\n'.format(m=msg))


def _output_message(_session, _message):
    """Output response to console and add to short term memory.

    :param _session: user chatbot session -- chatbot session object
    :param _message: message to console -- str
    """
    _session.remember_short_term(_message)
    print(_message)
    sys.stdout.flush()


# ---------------------------------------
# Chat Mode - Operations
# ---------------------------------------
def _find_right_bucket(length):
    """Find the proper bucket for an encoder input based on its length, use
    the smallest possible bucket.

    :param length: input sequence length -- int
    :return: bucket id -- int
    """
    return min([b for b in range(len(cb_config.BUCKETS))
                if cb_config.BUCKETS[b][0] >= length])


def _construct_response(output_logits, inv_dec_vocab):
    """Construct a response to the user's encoder input. Both 'greedy' and
    'beam search' sampling methods are available for use.

    :param output_logits: the outputs from sequence to sequence wrapper -- list
    :param inv_dec_vocab: inverted decoder vocab -- list
    :return: chatbot response -- str
    """
    # If user has set sample mode to 'beam search' Get the output response
    # using beam selection to determine the K best possible responses, takes
    # longer to process a response to user than argmax
    if cb_config.DECODE_MODE.lower() == 'beam':
        # Generate X possible responses from the model output logits
        possiblities = beam_search_decoder(output_logits, 3)

        # If beam search debug is enabled output all responses generated
        if cb_config.BEAM_DEBUG:
            print('***************')
            print('BEAM SEARCH DEBUG: Hal generated the three following '
                  'responses from beam search decoding:')
            for possibility in possiblities:
                r = possibility[0]
                # If EOS marker in sequence, slice the output list there
                if cb_config.EOS_ID in r:
                    r = r[:r.index(cb_config.EOS_ID)]
                print('\t' + ' '.join(
                    [tf.compat.as_str(inv_dec_vocab[output]) for output in r]))
            print('***************')

        # Get the response with the highest probability of being correct
        max_p = max(possiblities, key=lambda l: l[1])
        # Extract the output sequence from max probability sequence
        outputs = max_p[0]
    # Get the output response using greedy selection, the max output logit
    # for each logit in the output logits
    else:
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    # If EOS marker in sequence, slice the output list there
    if cb_config.EOS_ID in outputs:
        outputs = outputs[:outputs.index(cb_config.EOS_ID)]

    # Print out sentence corresponding to outputs IDs
    return ' '.join(
        [tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])


def beam_search_decoder(data, k):
    """Use beam search selection method for decoding.

    :param data: output logits -- nested list
    :param k: amount of predicted responses to return -- int
    :return: predicted responses -- nested list [[sequence, score], ...]
    """
    # Initialise response sequences
    p_seq = [[list(), 0.0]]
    # For each row in the output logits
    for r in data:
        # Initialise candidate list
        all_candidates = list()
        # Expand each current candidate
        for i in range(len(p_seq)):
            # Get sequence and score
            seq, score = p_seq[i]
            # Select k best word possibilities
            best_k = np.argsort(r)[-k:]
            # Explore k best word possibilities
            for j in best_k[0]:
                # Convert logit to probability so we only have + values
                odds = np.exp(r[0, j]) / (1 + np.exp(r[0, j]))
                # Add the selection to the sequence and add the log of the odds
                candidate = [seq + [j], score + math.log(odds)]
                # Add the selection candidate to the candidate list
                all_candidates.append(candidate)
        # Order all candidates by score
        ordered = sorted(all_candidates, key=lambda s: s[1])
        # Select k best responses to return
        p_seq = ordered[:k]
    return p_seq


if __name__ == '__main__':
    # Parse user input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices={'train', 'chat'}, default='train',
        help="Input '--mode' should be set, if ommitted defaults to 'train'.")
    args = parser.parse_args()

    # Process Data if dir does not exist already
    if not os.path.isdir(cb_config.PROCESSED_PATH):
        cb_data.main()
    print('Data ready!')

    # Create checkpoints folder
    cb_data.make_dir(cb_config.CPT_PATH)

    # Initialise model train or chat mode
    if args.mode == 'train':
        train()
    elif args.mode == 'chat':
        chat()
