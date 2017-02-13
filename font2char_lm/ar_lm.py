import os

import qnd
import tensorflow as tf


def def_ar_lm():
    qnd.add_flag('cell_size', type=int, default=8)
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
            tf.unstack(tf.transpose(batch.sequences['sentence'], [1, 0, 2])),
            sequence_length=batch.length,
            state_saver=batch,
            state_name=('c', 'h'))

        labels = batch.sequences['labels']
        print(labels)

        return loss

    return ar_lm
