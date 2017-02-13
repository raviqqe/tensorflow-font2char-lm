import font2char_lm
import numpy as np
import qnd
import qndex
import tensorflow as tf


def def_convert_text():
    get_chars = qndex.def_chars()

    def convert_text(string):
        char_indices = {char: index for index, char in enumerate(get_chars())}

        def convert(string):
            string = str(string)
            return (np.array([char_indices.get(char, qndex.nlp.UNKNOWN_INDEX)
                              for char in string],
                             dtype=np.int32),
                    len(string))

        document, length = tf.py_func(convert, [string], [tf.int32, tf.int32],
                                      name='convert_text')

        document.set_shape([length])

        return document

    return convert_text


def def_read_file():
    convert_text = def_convert_text()

    def read_file(filename_queue):
        key, value = tf.WholeFileReader().read(filename_queue)
        return key, convert_text(value)

    return read_file


char_lm = font2char_lm.def_char_lm()
read_file = def_read_file()


def main():
    qnd.train_and_evaluate(char_lm, read_file)


if __name__ == '__main__':
    main()
