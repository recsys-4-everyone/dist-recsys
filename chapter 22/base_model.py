# -*- coding: utf-8 -*-
import tensorflow as tf
import math
import sys


def input_fn():
    data = [
        {'label': 0, 'user_id': 'uid012', 'age': 18, 'item_id': 'item012', 'clicked_items_15d': ['item012', 'item345']},
        {'label': 0, 'user_id': 'uid012', 'age': 18, 'item_id': 'item012', 'clicked_items_15d': ['item012', 'item345']},
        {'label': 1, 'user_id': 'uid345', 'age': 18, 'item_id': 'item012', 'clicked_items_15d': ['item012', 'item345']}
    ]

    def generator():
        for instance in data:
            label = instance['label']
            features = {k: instance[k] for k in instance if k != 'label'}
            yield features, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=({'user_id': tf.string,
                       'age': tf.int64, 'item_id': tf.string, 'clicked_items_15d': tf.string},
                      tf.int64)
    )

    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(100)
    dataset = dataset.batch(32)

    return dataset


def embedding_size(category_num):
    return int(2 ** math.ceil(math.log2(category_num ** 0.25)))


# user
user_hash = tf.feature_column.categorical_column_with_hash_bucket(key='user_id', hash_bucket_size=10,
                                                                  dtype=tf.string)
user_embedding_size = embedding_size(100)
user_embedding = tf.feature_column.embedding_column(user_hash, user_embedding_size)

# age
raw_age = tf.feature_column.numeric_column(key='age', dtype=tf.int64)
boundaries = [18, 25, 36, 45, 55, 65, 80]
bucketized_age = tf.feature_column.bucketized_column(source_column=raw_age, boundaries=boundaries)
age_embedding_size = embedding_size(len(boundaries) + 1)
age_embedding = tf.feature_column.embedding_column(bucketized_age, age_embedding_size)

# item_id, clicked_items_15d
item_id = tf.feature_column.categorical_column_with_hash_bucket(key='item_id', hash_bucket_size=10, dtype=tf.string)
clicked_items_15d = tf.feature_column.categorical_column_with_hash_bucket(key='clicked_items_15d', hash_bucket_size=10,
                                                                          dtype=tf.string)
item_id_embedding, clicked_items_15d_embedding = tf.feature_column.shared_embedding_columns(
    [item_id, clicked_items_15d], dimension=2)


def deep_layers(layer, hidden_units, activation=None, name=''):
    layers = len(hidden_units)
    for i in range(layers - 1):
        num_hidden_units = hidden_units[i]
        layer = tf.layers.dense(layer, units=num_hidden_units, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                name=name + '_hidden_layer_{}'.format(
                                    str(num_hidden_units) + '_' + str(i)))
    num_hidden_units = hidden_units[layers - 1]
    layer = tf.layers.dense(layer, units=num_hidden_units, activation=activation,
                            kernel_initializer=tf.glorot_uniform_initializer(),
                            name=name + '_hidden_layer_{}'.format(str(num_hidden_units) + '_output'))
    return layer


def model_fn(features, labels, mode, params):
    learning_rate = params['learning_rate']
    fc_layers = params['fc_layers']
    inputs = tf.feature_column.input_layer(features, [user_embedding, age_embedding, item_id_embedding,
                                                      clicked_items_15d_embedding])
    logits = deep_layers(inputs, fc_layers, name='fc')
    probabilities = tf.sigmoid(logits)
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'ctr_auc': tf.metrics.auc(labels=labels,
                                      predictions=probabilities,
                                      num_thresholds=1000)
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metric_ops=metrics)
    else:
        global_step = tf.train.get_global_step()
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train_and_eval(net):
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(),
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(),
        steps=10,
        throttle_secs=180,
    )
    tf.estimator.train_and_evaluate(net, train_spec, eval_spec)


def main(unused):
    run_config = tf.estimator.RunConfig(**{
        'save_summary_steps': 10,
        'save_checkpoints_steps': 100,
        'keep_checkpoint_max': 5,
        'log_step_count_steps': 10
    })

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='./checkpoints',
        config=run_config,
        params={
            'learning_rate': 0.01,
            'fc_layers': [8, 4, 1]
        }
    )

    train_and_eval(model)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]])
