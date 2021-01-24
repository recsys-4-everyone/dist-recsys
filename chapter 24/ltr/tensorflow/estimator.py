# -*- coding: utf-8 -*-
import math
import tensorflow as tf
from ndcg import metrics_impl
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
        # shape: B (batch size) * concatenated embedding size
        user_embedding = fc.input_layer(features, [user_id, age, gender])

    with tf.name_scope('item'):
        item_buckets = 100
        # shape:
        # 训练:B * L (List size)
        # 线上: L, reshape to (1, L)
        item_id = features['item_id']
        if mode == tf.estimator.ModeKeys.PREDICT:
            item_id = tf.reshape(item_id, [1, -1])
        list_size = tf.shape(item_id)[1]
        item_id = tf.string_to_hash_bucket_fast(item_id, num_buckets=item_buckets)
        item_matrix = tf.get_variable(name='item_matrix',
                                      shape=(100, 16),
                                      initializer=tf.initializers.glorot_uniform())

        # shape: B * L * E (Embedding size)
        item_embedding = tf.nn.embedding_lookup(item_matrix,
                                                item_id,
                                                name='item_embedding')

    with tf.name_scope('history'):
        # shape: 训练: B * L * T (sequence length), 线上预测: 1 * T
        clicked_items = features['clicked_items_15d']
        # shape: 训练: B * L * T (sequence length), 线上预测: 1 * T
        clicked_mask = tf.cast(tf.not_equal(clicked_items, '0'), tf.bool)
        clicked_items = tf.string_to_hash_bucket_fast(clicked_items, num_buckets=item_buckets)
        # shape: 训练: B * L * T * E, 线上预测: 1 * T * E
        clicked_embedding = tf.nn.embedding_lookup(item_matrix,
                                                   clicked_items,
                                                   name='clicked_embedding')

    # if mode == tf.estimator.ModeKeys.PREDICT:
    # 复制 user_embedding 为 B * L * concatenated embedding size
    user_embedding = tf.expand_dims(user_embedding, 1)
    user_embedding = tf.tile(user_embedding, [1, list_size, 1])
    if mode == tf.estimator.ModeKeys.PREDICT:
        # 1 * T * E
        clicked_embedding = tf.expand_dims(clicked_embedding, 1)
        # 1 * L * T * E
        clicked_embedding = tf.tile(clicked_embedding, [1, list_size, 1, 1])
        # 1 * T
        clicked_mask = tf.expand_dims(clicked_mask, 1)
        # 1 * L * T
        clicked_mask = tf.tile(clicked_mask, [1, list_size, 1])

    # shape: B * L * E
    clicked_attention = attention(clicked_embedding,
                                  item_embedding,
                                  clicked_mask,
                                  [16, 8],
                                  'clicked_attention')

    fc_inputs = tf.concat([user_embedding, item_embedding, clicked_attention], axis=-1, name='fc_inputs')

    with tf.name_scope('predictions'):
        # B * L * 1
        logits = fc_layers(mode, net=fc_inputs, hidden_units=[64, 16, 1], dropout=0.3)
        logits = tf.squeeze(logits, axis=-1)  # B * L

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = tf.nn.softmax(logits, name='predictions')  # B * L (线上预测时 B = 1)
            predictions = {
                'probability': predictions
            }
            export_outputs = {
                'predictions': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                              export_outputs=export_outputs)
        else:
            relevance = tf.cast(labels, tf.float32)  # B * L
            soft_max = tf.nn.softmax(relevance, axis=-1)  # 1
            mask = tf.cast(relevance >= 0.0, tf.bool)
            loss = _masked_softmax_cross_entropy(logits=logits, labels=soft_max, mask=mask)  # 2

            if mode == tf.estimator.ModeKeys.EVAL:
                weights = tf.cast(mask, tf.float32)

                metrics = {}
                metrics.update(_ndcg(relevance, logits, weights=weights, name='ndcg'))

                for metric_name, op in metrics.items():
                    tf.summary.scalar(metric_name, op[1])
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  eval_metric_ops=metrics)
            else:
                global_step = tf.train.get_global_step()
                learning_rate = exponential_decay(global_step,
                                                  init_learning_rate,
                                                  decay_steps,
                                                  decay_rate)
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
                tf.summary.scalar('learning_rate', learning_rate)
                train_op = optimizer.minimize(loss=loss, global_step=global_step)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def attention(history, target, history_mask, hidden_units, name='attention_out'):
    """
    :param history: 训练: B * L * T * E, 线上预测: 1 * T * E
    :param target: 训练: B * L * E, 线上预测: 1 * L * E
    :param history_mask: 训练: B * L * T, 线上预测: 1 * T
    :param hidden_units: hidden units
    :param name: name
    :return: weighted sum tensor: 训练: B * L * E, 线上预测: 1 * L * E
    """
    # list_size: get L
    list_size = tf.shape(history)[1]
    # keys length: get T
    keys_length = tf.shape(history)[2]
    target_emb_size = target.get_shape()[-1]
    target_emb_tmp = tf.tile(target, [1, 1, keys_length])
    target = tf.reshape(target_emb_tmp, shape=[-1, list_size, keys_length, target_emb_size])
    net = tf.concat([history, history - target, target, history * target], axis=-1)
    for units in hidden_units:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    # net: B * L * T * hidden_units[-1]
    # attention_weight: B * L * T * 1
    attention_weight = tf.layers.dense(net, units=1, activation=None)
    scores = tf.transpose(attention_weight, [0, 1, 3, 2])  # B * L * 1 * T
    history_mask = tf.expand_dims(history_mask, axis=2)  # B * L * T --> B * L * 1 * T
    padding = tf.zeros_like(scores)  # B * L * 1 * T
    # mask 为 true 时使用 score, false 时使用 0
    scores = tf.where(history_mask, scores, padding)  # B * L * 1 * T
    outputs = tf.matmul(scores, history)  # [B * L * 1 * T] * [B * L * T * E] --> [B * L * 1 * E]
    outputs = tf.squeeze(outputs, axis=2, name=name)  # 去掉维度为 1 的 axis, B * L * E
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


def _ndcg(relevance, predictions, ks=(1, 4, 8, 20, None), weights=None, name='ndcg'):
    ndcgs = {}
    for k in ks:
        metric = metrics_impl.NDCGMetric('ndcg', topn=k, gain_fn=lambda label: tf.pow(2.0, label) - 1,
                                         rank_discount_fn=lambda rank: tf.math.log(2.) / tf.math.log1p(rank))

        with tf.name_scope(metric.name):
            per_list_ndcg, per_list_weights = metric.compute(relevance, predictions, weights)

        ndcgs.update({'{}_{}'.format(name, k): tf.metrics.mean(per_list_ndcg, per_list_weights)})

    return ndcgs


def _masked_softmax_cross_entropy(logits, labels, mask):
    """Softmax cross-entropy loss with masking."""
    padding = tf.ones_like(logits) * -2 ** 32
    logits = tf.where(mask, logits, padding)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
