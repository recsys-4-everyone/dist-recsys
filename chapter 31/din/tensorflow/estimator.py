# -*- coding: utf-8 -*-
import math
import tensorflow as tf
from tensorflow import feature_column as fc


def embedding_size(category_num):
    return int(2 ** math.ceil(math.log2(category_num ** 0.25)))


# user_id
user_id = fc.categorical_column_with_hash_bucket("user_id",
                                                 hash_bucket_size=10000)
user_id = fc.embedding_column(user_id,
                              dimension=embedding_size(10000))

# age
raw_age = fc.numeric_column('age',
                            default_value=0,
                            dtype=tf.int64)

boundaries = [18, 25, 36, 45, 55, 65, 80]
bucketized_age = fc.bucketized_column(source_column=raw_age,
                                      boundaries=boundaries)

age = fc.embedding_column(bucketized_age,
                          dimension=embedding_size(len(boundaries) + 1))

# gender
gender = fc.categorical_column_with_hash_bucket('gender',
                                                hash_bucket_size=100)
gender = fc.embedding_column(gender,
                             dimension=embedding_size(100))


def model_fn(features, labels, mode, params):
    init_learning_rate = params['learning_rate']
    decay_steps = params['decay_steps']
    decay_rate = params['decay_rate']

    with tf.name_scope('user'):
        # shape: B (batch size)
        user_embedding = fc.input_layer(features, [user_id, age, gender])

    with tf.name_scope('item'):
        item_buckets = 100
        item_id = features['item_id']
        item_id = tf.reshape(item_id, [-1, 1])
        list_size = tf.shape(item_id)[0]
        item_id = tf.string_to_hash_bucket_fast(item_id, num_buckets=item_buckets)
        # if matrix is huge, it can be distributed
        # item_matrix = tf.get_variable(name='item_matrix',
        #                               shape=(100, 16),
        #                               initializer=tf.initializers.glorot_uniform())
        if mode != tf.estimator.ModeKeys.PREDICT:
            ps_num = len(params['tf_config']['cluster']['ps'])
            item_matrix = tf.get_variable(name='item_matrix',
                                          shape=(100, 16),
                                          initializer=tf.initializers.glorot_uniform(),
                                          partitioner=tf.fixed_size_partitioner(num_shards=ps_num)) #1
        else:
            item_matrix = tf.get_variable(name='item_matrix',
                                          shape=(100, 16),
                                          initializer=tf.initializers.glorot_uniform())

        item_embedding = tf.nn.embedding_lookup(item_matrix,
                                                item_id,
                                                name='item_embedding')
        item_embedding = tf.squeeze(item_embedding, axis=1)

    with tf.name_scope('history'):
        # shape: B * T (sequence length)
        clicked_items = features['clicked_items_15d']
        clicked_mask = tf.cast(tf.not_equal(clicked_items, '0'), tf.bool)
        clicked_items = tf.string_to_hash_bucket_fast(clicked_items, num_buckets=item_buckets)
        # shape: B * T * E
        clicked_embedding = tf.nn.embedding_lookup(item_matrix,
                                                   clicked_items,
                                                   name='clicked_embedding')

    if mode == tf.estimator.ModeKeys.PREDICT:
        user_embedding = tf.tile(user_embedding, [list_size, 1])
        clicked_embedding = tf.tile(clicked_embedding, [list_size, 1, 1])
        clicked_mask = tf.tile(clicked_mask, [list_size, 1])

    # shape: B * E
    clicked_attention = attention(clicked_embedding,
                                  item_embedding,
                                  clicked_mask,
                                  [16, 8],
                                  'clicked_attention')

    fc_inputs = tf.concat([user_embedding, item_embedding, clicked_attention], axis=-1, name='fc_inputs')

    with tf.name_scope('predictions'):
        logits = fc_layers(mode, net=fc_inputs, hidden_units=[64, 16, 1], dropout=0.3)
        predictions = tf.sigmoid(logits, name='predictions')

        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = tf.reshape(labels, [-1, 1])
            loss = tf.losses.sigmoid_cross_entropy(labels, logits)
            if mode == tf.estimator.ModeKeys.EVAL:
                metrics = {
                    'auc': tf.metrics.auc(labels=labels,
                                          predictions=predictions,
                                          num_thresholds=500)
                }
                for metric_name, op in metrics.items():
                    tf.summary.scalar(metric_name, op[1])
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  eval_metric_ops=metrics)
            else:
                global_step = tf.train.get_global_step()
                learning_rate = exponential_decay(global_step, init_learning_rate, decay_steps, decay_rate)
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
                tf.summary.scalar('learning_rate', learning_rate)
                train_op = optimizer.minimize(loss=loss, global_step=global_step)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            predictions = {
                'probability': tf.reshape(predictions, [1, -1])
            }
            export_outputs = {
                'predictions': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                              export_outputs=export_outputs)


def attention(history, target, history_mask, hidden_units, name='attention_out'):
    keys_length = tf.shape(history)[1]
    target_emb_size = target.get_shape()[-1]
    target_emb_tmp = tf.tile(target, [1, keys_length])
    target = tf.reshape(target_emb_tmp, shape=[-1, keys_length, target_emb_size])
    net = tf.concat([history, history - target, target, history * target], axis=-1)
    for units in hidden_units:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    attention_weight = tf.layers.dense(net, units=1, activation=None)
    scores = tf.transpose(attention_weight, [0, 2, 1])
    history_masks = tf.expand_dims(history_mask, axis=1)
    padding = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(history_masks, scores, padding)
    scores = tf.nn.softmax(scores)
    outputs = tf.matmul(scores, history)
    outputs = tf.reduce_sum(outputs, 1, name=name)
    return outputs


def fc_layers(mode, net, hidden_units, dropout=0.0, activation=None, name='fc_layers'):
    layers = len(hidden_units)
    for i in range(layers - 1):
        num = hidden_units[i]
        net = tf.layers.dense(net, units=num, activation=tf.nn.relu,
                              kernel_initializer=tf.initializers.he_uniform(),
                              name=name + '_hidden_{}'.format(str(num) + '_' + str(i)))
        net = tf.layers.dropout(inputs=net, rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
    num = hidden_units[layers - 1]
    net = tf.layers.dense(net, units=num, activation=activation,
                          kernel_initializer=tf.initializers.glorot_uniform(),
                          name=name + '_hidden_{}'.format(str(num)))
    return net


def exponential_decay(global_step, learning_rate=0.1, decay_steps=10000, decay_rate=0.9):
    return tf.train.exponential_decay(learning_rate=learning_rate,
                                      global_step=global_step,
                                      decay_steps=decay_steps,
                                      decay_rate=decay_rate,
                                      staircase=False)
