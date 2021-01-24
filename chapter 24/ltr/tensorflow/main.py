# -*- coding: utf-8 -*-

import os
from data import input_fn
from estimator import model_fn
import tensorflow as tf


def main(_):
    cpu = os.cpu_count()
    session_config = tf.ConfigProto(
        device_count={'GPU': 0,
                      'CPU': cpu},
        inter_op_parallelism_threads=cpu // 2,
        intra_op_parallelism_threads=cpu // 2,
        device_filters=[],
        allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(**{
        'save_summary_steps': 100,
        'save_checkpoints_steps': 1000,
        'keep_checkpoint_max': 10,
        'log_step_count_steps': 100
    }).replace(session_config=session_config)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='/home/axing/ltr/checkpoints/ltr',
        config=run_config,
        params={
            'decay_rate': 0.9,
            'decay_steps': 10000,
            'learning_rate': 0.1
        }
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(mode='train', pattern='/home/axing/ltr/dataset/*'))
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(mode='eval', pattern='/home/axing/ltr/dataset/*'),
        steps=100,
        throttle_secs=60
    )
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
