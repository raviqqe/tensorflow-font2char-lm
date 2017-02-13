import qnd
import tensorflow as tf


def def_ar_lm():
    qnd.add_flag('cell_size', type=int, default=8)
    qnd.add_flag('num_unroll', type=int, default=16)

    def ar_lm(key, sequence, labels, *, char_embeddings):
        cell = tf.contrib.rnn.LayerNormalBasicLSTMCell(qnd.FLAGS.cell_size)

        batch = tf.batch_sequences_with_states(
            key,
            input_sequneces={
                'sequence': tf.gather(char_embeddings, sequence),
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
            tf.split(1, qnd.FLAGS.num_unroll, batch.sequences['sequence']),
            sequence_length=batch.length,
            state_saver=batch,
            state_name=('c', 'h'))

        # TODO: Calculate loss.
        labels = batch.sequences["labels"]
        return loss

    return ar_lm
