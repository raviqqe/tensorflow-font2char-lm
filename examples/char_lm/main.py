import font2char_lm
import numpy as np
import qnd
import qndex
import tensorflow as tf


def def_convert_text():
    get_chars = qndex.nlp.def_chars()

    def convert_text(string):
        char_indices = {char: index for index, char in enumerate(get_chars())}

        def convert(string):
            sentence = np.array(
                [char_indices['<s>'],
                 *[char_indices.get(char, qndex.nlp.UNKNOWN_INDEX)
                   for char in str(string)]],
                dtype=np.int32)

            return (sentence,
                    np.array([*sentence[1:], char_indices['</s>']],
                             dtype=sentence.dtype),
                    len(sentence))

        sentence, labels, length = tf.py_func(
            convert, [string], [tf.int32, tf.int32, tf.int32],
            name='convert_text')

        length.set_shape([])

        return tf.reshape(sentence, [length]), tf.reshape(labels, [length])

    return convert_text


def def_read_file():
    convert_text = def_convert_text()

    def read_file(filename_queue):
        key, value = tf.WholeFileReader().read(filename_queue)
        sentence, labels = convert_text(value)
        return {'key': key, 'sentence': sentence}, {'labels': labels}

    return read_file


char_lm = font2char_lm.def_char_lm()
read_file = def_read_file()
train_and_evaluate = qnd.def_train_and_evaluate(batch_inputs=False)


def main():
    train_and_evaluate(char_lm, read_file)


if __name__ == '__main__':
    main()
