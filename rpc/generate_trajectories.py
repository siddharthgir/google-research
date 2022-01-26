# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script for training and evaluating RPC agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import time

from absl import app
from absl import flags
import math
from absl import logging
import gin
import numpy as np
import rpc_agent
import rpc_utils
from six.moves import range
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents import data_converter
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

flags.DEFINE_string('policy_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Directory containing saved policy')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS



@gin.configurable
def generate_trajectories(
    policy_dir,
    rb_save_dir="/home/sgirdhar/rpc/rpc_rb_buffers_1/",
    env_name='HalfCheetah-v2',
    actor_fc_layers=(),
    latent_dim=10,
    clip_mean=30.0,
    clip_max_stddev=10.0,
    clip_min_stddev=0.1,
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    trajectories_count=10000,
    num_eval_episodes=30,
    replay_buffer_capacity=10000):

    env = suite_gym.load(env_name)
    tf_env = tf_py_environment.TFPyEnvironment(env)

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    

    def _activation(t):
      t1, t2 = tf.split(t, 2, axis=1)
      low = -np.inf if clip_mean is None else -clip_mean
      high = np.inf if clip_mean is None else clip_mean
      t1 = rpc_utils.squash_to_range(t1, low, high)

      if clip_min_stddev is None:
        low = -np.inf
      else:
        low = tf.math.log(tf.exp(clip_min_stddev) - 1.0)
      if clip_max_stddev is None:
        high = np.inf
      else:
        high = tf.math.log(tf.exp(clip_max_stddev) - 1.0)
      t2 = rpc_utils.squash_to_range(t2, low, high)
      return tf.concat([t1, t2], axis=1)

    encoder_net = tf.keras.Sequential([
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dense(
              tfp.layers.IndependentNormal.params_size(latent_dim),
              activation=_activation,
              kernel_initializer='glorot_uniform'),
          tfp.layers.IndependentNormal(latent_dim),
      ])

    actor_net = rpc_utils.ActorNet(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        encoder=encoder_net,
        predictor=None,
        fc_layers=actor_fc_layers)

    critic_net = rpc_utils.CriticNet(
        (observation_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

    tf_agent = rpc_agent.RpAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        actor_optimizer=None,
        alpha_optimizer=None,
        critic_network=critic_net,
        critic_optimizer=None)

    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    policy_checkpointer = common.Checkpointer(
    policy = eval_policy,
    ckpt_dir=policy_dir)

    policy_checkpointer.initialize_or_restore()

    

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=tf_agent.collect_data_spec,
      batch_size=tf_env.batch_size,
      max_length=replay_buffer_capacity)

    rb_checkpointer = common.Checkpointer(
        ckpt_dir=rb_save_dir,
        max_to_keep=1,
        replay_buffer=replay_buffer)

    rb_checkpointer.initialize_or_restore()

    class counter(object):
      def __init__(self):
        self.count = 0

      def printCount(self,x):
        self.count += 1
        if self.count%10000 == 0:
          print("on iteration",self.count)

    c = counter()
    collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env,
    eval_policy,
    observers=[replay_buffer.add_batch,c.printCount],
    num_steps=trajectories_count)

    collect_driver.run()
    rb_checkpointer.save(0)







def main(_):
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  tf.config.set_visible_devices([], 'GPU')

  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  if 'xm_parameters' in FLAGS and FLAGS.xm_parameters:
    hparams = json.loads(FLAGS.xm_parameters)
    with gin.unlock_config():
      for (key, value) in hparams.items():
        print('Setting: %s = %s' % (key, value))
        gin.bind_parameter(key, value)

  policy_dir = FLAGS.policy_dir
  generate_trajectories(policy_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('policy_dir')

  app.run(main)
