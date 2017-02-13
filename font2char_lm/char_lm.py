import qnd
import qndex
import extenteten as ex
import tensorflow as tf

from .ar_lm import def_ar_lm


def def_char_lm():
    get_chars = qndex.nlp.def_chars()
    qnd.add_flag('char_embedding_size', type=int, default=100)
    ar_lm = def_ar_lm()

    def char_lm(key, sentence, labels):
        return ar_lm(
            key,
            sentence,
            labels,
            char_embeddings=ex.embeddings(
                id_space_size=len(get_chars()),
                embedding_size=qnd.FLAGS.char_embedding_size,
                name='char_embeddings'))

    return char_lm
