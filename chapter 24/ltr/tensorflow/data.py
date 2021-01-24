# -*- coding: utf-8 -*-

import tensorflow as tf


def get_example_fmt():
    context_fmt = {}
    sequence_fmt = {}

    # 注意每个特征的类型哦
    context_fmt['user_id'] = tf.FixedLenFeature([], tf.string)
    context_fmt['age'] = tf.FixedLenFeature([], tf.int64)
    context_fmt['gender'] = tf.FixedLenFeature([], tf.string)

    context_fmt['item_id'] = tf.VarLenFeature(tf.string)
    context_fmt['relevance'] = tf.VarLenFeature(tf.int64)

    # 此特征放在了 sequence_fmt 中，表明了它是一个数组的数组
    sequence_fmt['clicked_items_15d'] = tf.VarLenFeature(tf.string)

    return context_fmt, sequence_fmt


def parse_fn(example):
    context_fmt, sequence_fmt = get_example_fmt()
    context, sequence = tf.parse_single_sequence_example(example, context_fmt, sequence_fmt)

    parsed = context
    parsed.update(sequence)
    parsed['item_id'] = tf.sparse.to_dense(parsed['item_id'], '0')
    parsed['clicked_items_15d'] = tf.sparse.to_dense(parsed['clicked_items_15d'], '0')
    relevance = parsed.pop('relevance')
    relevance = tf.sparse.to_dense(relevance, -2 ** 32)  # 1
    features = parsed
    return features, relevance


# pad 返回的数据格式与形状必须与 parse_fn 的返回值完全一致。
def padded_shapes_and_padding_values():
    context_fmt, sequence_fmt = get_example_fmt()

    padded_shapes = {}
    padding_values = {}

    padded_shapes['user_id'] = []
    padded_shapes['age'] = []
    padded_shapes['gender'] = []
    padded_shapes['item_id'] = [None]
    padded_shapes['clicked_items_15d'] = [None, None]

    padding_values['user_id'] = '0'
    padding_values['age'] = tf.constant(0, dtype=tf.int64)
    padding_values['gender'] = '0'
    padding_values['item_id'] = '0'
    padding_values['clicked_items_15d'] = '0'

    padded_shapes = (padded_shapes, [None])
    padding_values = (padding_values, -2 ** 32)  # 2
    return padded_shapes, padding_values


def input_fn(mode, pattern, epochs=1, batch_size=512, num_parallel_calls=1):
    padded_shapes, padding_values = padded_shapes_and_padding_values()
    files = tf.data.Dataset.list_files(pattern)
    data_set = files.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=8,
            sloppy=True
        )
    )  # 1
    data_set = data_set.apply(tf.data.experimental.ignore_errors())
    data_set = data_set.map(map_func=parse_fn,
                            num_parallel_calls=num_parallel_calls)  # 2

    if mode == 'train':
        data_set = data_set.shuffle(buffer_size=10000)  # 3.1
        data_set = data_set.repeat(epochs)  # 3.2
    data_set = data_set.padded_batch(batch_size,
                                     padded_shapes=padded_shapes,
                                     padding_values=padding_values)

    data_set = data_set.prefetch(buffer_size=1)  # 4
    return data_set
