import os

import extenteten as ex
import qnd
import tensorflow as tf


def def_ar_lm():
    qnd.add_flag('cell_size', type=int, default=128)
    qnd.add_flag('num_unroll', type=int, default=16)
    qnd.add_flag('batch_size', type=int, default=64)
    qnd.add_flag('num_batch_threads', type=int, default=os.cpu_count())
    qnd.add_flag('batch_queue_capacity', type=int, default=1024)

    def ar_lm(key, sentence, labels, *, char_embeddings):
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(qnd.FLAGS.cell_size)

        batch = tf.contrib.training.batch_sequences_with_states(
            key,
            input_sequences={
                'sentence': tf.gather(char_embeddings, sentence),
                'labels': labels,
            },
            input_context={},
            input_length=None,
            initial_states={
                'c': tf.zeros([cell.state_size.c], tf.float32),
                'h': tf.zeros([cell.state_size.h], tf.float32),
            },
            num_unroll=qnd.FLAGS.num_unroll,
            batch_size=qnd.FLAGS.batch_size,
            num_threads=qnd.FLAGS.num_batch_threads,
            capacity=qnd.FLAGS.batch_queue_capacity)

        outputs, _ = tf.nn.state_saving_rnn(
            cell,
            tf.unstack(batch.sequences['sentence'], axis=1),
            sequence_length=batch.length,
            state_saver=batch,
            state_name=('c', 'h'))

        logits = batch_linear(outputs, ex.static_shape(char_embeddings)[0])
        labels = batch.sequences['labels']

        loss = sequence_labeling_loss(logits, labels, batch.length)

        return (
            {
                'key': key,
                'labels': (tf.argmax(logits, axis=2) *
                           tf.sequence_mask(batch.length, dtype=tf.int64)),
            },
            loss,
            ex.minimize(loss),
        )

    return ar_lm


def batch_linear(h, output_size):
    assert ex.static_rank(h) == 3

    shape = ex.static_shape(h)
    return (
        tf.batch_matmul(
            h,
            tf.tile(tf.expand_dims(ex.variable([shape[2], output_size]), 0),
                    [shape[0], 1, 1]))
        + ex.variable([output_size]))


def sequence_labeling_loss(logits, labels, sequence_length=None):
    assert ex.static_rank(logits) == 3
    assert ex.static_rank(labels) == 2

    losses = tf.reshape(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.reshape(logits, [-1, ex.static_shape(logits)[-1]]),
            tf.reshape(labels, [-1])),
        [-1, *ex.static_shape(labels)[1:]])

    if sequence_length == None:
        return tf.reduce_mean(losses)

    mask = tf.sequence_mask(sequence_length, dtype=losses.dtype)

    return tf.reduce_sum(losses * mask) / tf.reduce_sum(mask)
