# -*- coding: utf-8 -*-
import tensorflow as tf
from estimator import model_fn


def export_model():
    def serving_input_receiver_fn():
        receiver_tensors = \
            {'user_id': tf.placeholder(dtype=tf.string,
                                       shape=(None, None),
                                       name='user_id'),
             'age': tf.placeholder(dtype=tf.int64,
                                   shape=(None, None),
                                   name='age'),
             'gender': tf.placeholder(dtype=tf.string,
                                      shape=(None, None),
                                      name='gender'),
             'item_id': tf.placeholder(dtype=tf.string,
                                       shape=(None, None),
                                       name='item_id'),
             'clicked_items_15d': tf.placeholder(dtype=tf.string,
                                                 shape=(None, None),
                                                 name='clicked_items_15d')
             }

        return tf.estimator.export.build_raw_serving_input_receiver_fn(receiver_tensors)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='/home/axing/ltr/checkpoints/ltr',
        params={
            'decay_rate': 0.9,
            'decay_steps': 50000,
            'learning_rate': 0.1
        }
    )

    model.export_savedmodel('/home/axing/ltr/savers/ltr', serving_input_receiver_fn())


def main(_):
    export_model()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
