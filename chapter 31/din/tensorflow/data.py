# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.compat.v1 import data
from tensorflow.compat.v1.data import experimental
import os


def get_example_fmt():
    example_fmt = dict()

    example_fmt['label'] = tf.FixedLenFeature([], tf.int64)
    example_fmt['user_id'] = tf.FixedLenFeature([], tf.string)
    example_fmt['age'] = tf.FixedLenFeature([], tf.int64)
    example_fmt['gender'] = tf.FixedLenFeature([], tf.string)
    example_fmt['item_id'] = tf.FixedLenFeature([], tf.string)
    # 此特征长度不固定
    example_fmt['clicked_items_15d'] = tf.VarLenFeature(tf.string)

    return example_fmt


def padded_shapes_and_padding_values():
    example_fmt = get_example_fmt()

    padded_shapes = {}
    padding_values = {}

    for f_name, f_fmt in example_fmt.items():
        if 'label' == f_name:
            continue
        if isinstance(f_fmt, tf.FixedLenFeature):
            padded_shapes[f_name] = []
        elif isinstance(f_fmt, tf.VarLenFeature):
            padded_shapes[f_name] = [None]
        else:
            raise NotImplementedError('feature {} feature type error.'.format(f_name))

        if f_fmt.dtype == tf.string:
            value = '0'
        elif f_fmt.dtype == tf.int64:
            value = 0
        elif f_fmt.dtype == tf.float32:
            value = 0.0
        else:
            raise NotImplementedError('feature {} data type error.'.format(f_name))

        padding_values[f_name] = tf.constant(value, dtype=f_fmt.dtype)

    padded_shapes = (padded_shapes, [])
    padding_values = (padding_values, tf.constant(0, tf.int64))
    return padded_shapes, padding_values


def parse_fn(example):
    example_fmt = get_example_fmt()
    parsed = tf.parse_single_example(example, example_fmt)
    # VarLenFeature 解析的特征是 Sparse 的，需要转成 Dense 便于操作
    parsed['clicked_items_15d'] = tf.sparse.to_dense(parsed['clicked_items_15d'], '0')
    label = parsed['label']
    parsed.pop('label')
    features = parsed
    return features, label


def input_fn(mode, pattern, epochs=1, batch_size=512, num_workers=None, worker_index=None):
    cpus = os.cpu_count()
    padded_shapes, padding_values = padded_shapes_and_padding_values()
    files = tf.data.Dataset.list_files(pattern)
    if num_workers and num_workers > 0 and worker_index > -1: #1
        files = files.shard(num_workers, worker_index)
    data_set = files.apply(experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=cpus, sloppy=True))
    data_set = data_set.apply(experimental.ignore_errors())
    data_set = data_set.map(map_func=parse_fn, num_parallel_calls=cpus)
    if mode == 'train':
        data_set = data_set.shuffle(buffer_size=10000)
        data_set = data_set.repeat(epochs)
    data_set = data_set.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
    data_set = data_set.prefetch(buffer_size=experimental.AUTOTUNE)
    return data_set