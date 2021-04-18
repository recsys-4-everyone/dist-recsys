# -*- coding: utf-8 -*-

import os
import json
import args
import argparse
from data import input_fn
from estimator import model_fn
from tensorflow.compat.v1 import app
from tensorflow.compat.v1 import logging
from tensorflow.compat.v1 import estimator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1.distribute import experimental

parser = argparse.ArgumentParser()
args.add_arguments(parser)
flags, un_parsed = parser.parse_known_args()


def _tf_config(_flags):
    tf_config = dict()
    ps = ['localhost:2220']
    chief = ['localhost:2221']
    worker = ['localhost:2222']
    evaluator = ['localhost:2223']

    cluster = {
        'ps': ps,
        'chief': chief,
        'worker': worker,
        'evaluator': evaluator
    }

    task = {
        'type': _flags.type,
        'index': _flags.index
    }

    tf_config['cluster'] = cluster
    tf_config['task'] = task

    if _flags.type == 'chief':
        _flags.__dict__['worker_index'] = 0
    elif _flags.type == 'worker':
        _flags.__dict__['worker_index'] = 1

    _flags.__dict__['num_workers'] = len(worker) + len(chief)
    
    _flags.__dict__['device_filters'] = ["/job:ps", f"/job:{_flags.type}/task:{_flags.index}"]

    return tf_config


def main(_):
    cpu = os.cpu_count()
    tf_config = _tf_config(flags) #1
    # 分布式需要 TF_CONFIG 环境变量
    os.environ['TF_CONFIG'] = json.dumps(tf_config) #2
    session_config = ConfigProto(
        device_count={'CPU': cpu},
        inter_op_parallelism_threads=cpu // 2,
        intra_op_parallelism_threads=cpu // 2,
        device_filters=flags.device_filters,
        allow_soft_placement=True)
    strategy = experimental.ParameterServerStrategy()
    run_config = estimator.RunConfig(**{
        'save_summary_steps': 100,
        'save_checkpoints_steps': 1000,
        'keep_checkpoint_max': 10,
        'log_step_count_steps': 100,
        'train_distribute': strategy,
        'eval_distribute': strategy,
    }).replace(session_config=session_config)

    model = estimator.Estimator(
        model_fn=model_fn,
        model_dir='/home/axing/din/checkpoints/din', #实际应用中是分布式文件系统
        config=run_config,
        params={
            'tf_config': tf_config,
            'decay_rate': 0.9,
            'decay_steps': 10000,
            'learning_rate': 0.1
        }
    )

    train_spec = estimator.TrainSpec(
        input_fn=lambda: input_fn(mode='train',
                                  num_workers=flags.num_workers,
                                  worker_index=flags.worker_index,
                                  pattern='/home/axing/din/dataset/*'), #3
        max_steps=1000  #4
    )

    # 这里就假设验证集和训练集地址一样了，实际应用中是肯定不一样的。
    eval_spec = estimator.EvalSpec(
        input_fn=lambda: input_fn(mode='eval', pattern='/home/axing/din/dataset/*'),
        steps=100,  # 每次验证 100 个 batch size 的数据
        throttle_secs=60  # 每隔至少 60 秒验证一次
    )
    estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main=main)
