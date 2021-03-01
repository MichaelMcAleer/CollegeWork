# ---------------------------------------------------------------
# Natural Language Processing
# Assignment 3 - Dialogue System
# Michael McAleer R00143621
#
# >>> Improvements Made:
# - Added dropout to each layer in LSTM nueral network to reduce overfitting
# - Changed optimiser to Adam Optimiser, changed learning rate and added decay
# - Added beam-search decoder
# ---------------------------------------------------------------
import numpy as np
import tensorflow as tf
import time

import cb_config


class ChatBotModel:
    """Chatbot model object."""

    def __init__(self, forward_only, batch_size):
        """Session parameters to set on initialisation.

        :param forward_only: if chat or evaluate mode is enabled -- bool
        :param batch_size: model batch size --  int
        """
        print('Initialize new model')
        self.fw_only = forward_only
        self.batch_size = batch_size

    def _create_placeholders(self):
        """Create encoder, decoder and mask input placeholders for model."""
        print('Create placeholders')
        self.encoder_inputs = [
            tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
            for i in range(cb_config.BUCKETS[-1][0])]
        self.decoder_inputs = [
            tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
            for i in range(cb_config.BUCKETS[-1][1] + 1)]
        self.decoder_masks = [
            tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
            for i in range(cb_config.BUCKETS[-1][1] + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <GO> symbol)
        self.targets = self.decoder_inputs[1:]

    def _inference(self):
        """Create model layers (with dropout wrappers) and set softmax loss
        function."""
        print('Create inference')
        # Set output projection value for sampled softmax
        if 0 < cb_config.NUM_SAMPLES < cb_config.DEC_VOCAB:
            w = tf.get_variable('proj_w',
                                [cb_config.HIDDEN_SIZE, cb_config.DEC_VOCAB])
            b = tf.get_variable('proj_b', [cb_config.DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(
                weights=tf.transpose(w), biases=b, inputs=logits,
                labels=labels, num_sampled=cb_config.NUM_SAMPLES,
                num_classes=cb_config.DEC_VOCAB)

        # Set model loss function
        self.softmax_loss_function = sampled_loss

        # Creation of the rnn cell with dropout if not in chat (forward pass
        # only) mode
        def rnn_cell(fw_only):
            single_cell = tf.contrib.rnn.GRUCell(cb_config.HIDDEN_SIZE)
            if fw_only:
                single_cell = tf.contrib.rnn.DropoutWrapper(
                    single_cell, input_keep_prob=1.0,
                    output_keep_prob=cb_config.DROPOUT)
            return single_cell

        # single_cell = tf.contrib.rnn.GRUCell(cb_config.HIDDEN_SIZE)
        self.cell = tf.contrib.rnn.MultiRNNCell(
            [rnn_cell(self.fw_only) for _ in range(cb_config.NUM_LAYERS)])

    def _create_loss(self):
        """Create model loss function."""
        print('Creating loss... \nIt might take a couple of minutes depending '
              'on how many buckets you have.')
        start = time.time()

        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            setattr(tf.contrib.rnn.GRUCell, '__deepcopy__',
                    lambda self, _: self)
            setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__',
                    lambda self, _: self)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, self.cell,
                num_encoder_symbols=cb_config.ENC_VOCAB,
                num_decoder_symbols=cb_config.DEC_VOCAB,
                embedding_size=cb_config.HIDDEN_SIZE,
                output_projection=self.output_projection,
                feed_previous=do_decode)

        # If chat or evaluation mode
        if self.fw_only:
            # Get mode outputs and losses, decode mode is True
            self.outputs, self.losses = (
                tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs,
                    self.targets, self.decoder_masks, cb_config.BUCKETS,
                    lambda x, y: _seq2seq_f(x, y, True),
                    softmax_loss_function=self.softmax_loss_function))
            # If we use output projection, we need to project outputs for
            # decoding.
            if self.output_projection:
                for bucket in range(len(cb_config.BUCKETS)):
                    self.outputs[bucket] = [
                        tf.matmul(output, self.output_projection[0]) +
                        self.output_projection[1] for output in
                        self.outputs[bucket]]
        else:
            # Else in training mode, decode mode is False
            self.outputs, self.losses = (
                tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs,
                    self.targets, self.decoder_masks, cb_config.BUCKETS,
                    lambda x, y: _seq2seq_f(x, y, False),
                    softmax_loss_function=self.softmax_loss_function))
        print('Time:', time.time() - start)

    def _create_optimizer(self):
        """Create model optimiser, valid options are SGD and Adam."""
        print('Create optimizer... \nIt might take a couple of minutes '
              'depending on how many buckets you have.')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                           name='global_step')
            # Set optimiser by user-defined conifg options, we don't need an
            # optimiser if we are in chat or evaluation mode
            if not self.fw_only:
                if cb_config.OPTIMISER.lower() == 'sgd':
                    self.optimizer = tf.train.GradientDescentOptimizer(
                        cb_config.LR)
                else:
                    self.optimizer = tf.train.AdamOptimizer(
                        learning_rate=cb_config.LR,
                        beta1=0.9, beta2=0.999, epsilon=1e-08)

                trainables = tf.trainable_variables()
                # Set the model gradient norms and optimiser gradients
                self.gradient_norms = list()
                self.train_ops = list()
                start = time.time()
                # Configure gradient normalisation to avoid exploding gradients
                # during training in later training steps
                for bucket in range(len(cb_config.BUCKETS)):
                    clipped_grads, norm = tf.clip_by_global_norm(
                        tf.gradients(self.losses[bucket], trainables),
                        cb_config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(
                        self.optimizer.apply_gradients(
                            zip(clipped_grads, trainables),
                            global_step=self.global_step))
                    print('Creating opt for bucket {} took {} seconds'.format(
                        bucket, time.time() - start))
                    start = time.time()

    def build_graph(self):
        """Contstruct the model graph in its entirety."""
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._create_optimizer()
