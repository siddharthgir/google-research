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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time

import pdb 

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
from tf_agents.networks import utils


flags.DEFINE_string('trained_agent_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Directory containing saved policy')
flags.DEFINE_string('eval_rb_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Directory containing saved policy')
flags.DEFINE_string('trained_predictor_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Directory containing saved policy')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS


def get_replay_buffer(rb_checkpoint_dir,data_spec,batch_size,max_length):
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=tf_agent.collect_data_spec,
      batch_size=tf_env.batch_size,
      max_length=replay_buffer_capacity)

  rb_checkpointer = common.Checkpointer(
    replay_buffer = replay_buffer,
    ckpt_dir=rb_checkpoint_dir)

  rb_checkpointer.initialize_or_restore()
  return replay_buffer


def get_agent(agent_dir,
  env_name='HalfCheetah-v2',
  latent_dim=10,
  predictor_num_layers=2,
  actor_fc_layers=(),
  clip_mean=30.0,
  clip_max_stddev=10.0,
  clip_min_stddev=0.1,
  critic_obs_fc_layers=None,
  critic_action_fc_layers=None,
  critic_joint_fc_layers=(256, 256)
  ):

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

  def _custom_activation(t):
    low = -np.inf if clip_mean is None else -clip_mean
    high = np.inf if clip_mean is None else clip_mean
    t = rpc_utils.squash_to_range(t, low, high)
    return t


  encoder_net = tf.keras.Sequential([
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dense(
              tfp.layers.IndependentNormal.params_size(latent_dim ),
              activation=_activation,
              kernel_initializer='glorot_uniform'),
          tfp.layers.IndependentNormal(latent_dim),
      ])

    
  """
  obs_input = tf.keras.layers.Input(observation_spec.shape)
  action_input = tf.keras.layers.Input(action_spec.shape)

  z = encoder_net(obs_input)
  z = tf.stop_gradient(z)
  za = tf.concat([z, action_input], axis=1)
  za_input = tf.keras.layers.Input(za.shape[1])
  loc_scale = tf.keras.Sequential(
    predictor_num_layers * [tf.keras.layers.Dense(256, activation='relu')] + [  # pylint: disable=line-too-long
    tf.keras.layers.Dense(
    tfp.layers.IndependentNormal.params_size(latent_dim),
    activation=_activation,
    kernel_initializer='zeros'),
    ])(za_input)

  combined_loc_scale = tf.concat([
    loc_scale[:, :latent_dim] + za_input[:, :latent_dim],
    loc_scale[:, latent_dim:]],axis=1)
        
  dist = tfp.layers.IndependentNormal(latent_dim)(combined_loc_scale)
  output = tf.keras.Model(inputs=za_input, outputs=dist)(za)

  predictor_net = tf.keras.Model(inputs=(obs_input, action_input),outputs=output)
  """
  batch_size=1
  obs_input = tf.keras.layers.Input((10,observation_spec.shape[0]),batch_size=batch_size)
  action_input = tf.keras.layers.Input((10,action_spec.shape[0]),batch_size=batch_size)
  obs = tf.reshape(obs_input,(batch_size*10,observation_spec.shape[0]))
  z = encoder_net(obs)
  z = tf.stop_gradient(z)
  z = tf.reshape(z,(batch_size,10,latent_dim))
  print(z.shape)
  za = tf.concat([z,action_input],axis=2)
  za_input = tf.keras.layers.Input((za.shape[1],za.shape[2]),batch_size=batch_size)
  lstm_model = tf.keras.layers.LSTM(64,stateful=False,return_sequences=False)(za_input)
  loc_scale = tf.keras.Sequential(
      2 * [tf.keras.layers.Dense(256, activation='relu')]+[tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(latent_dim),activation=_activation,kernel_initializer='zeros')])(lstm_model)
  combined_loc_scale = tf.concat([
      loc_scale[:, :latent_dim] + za_input[:, -1,:latent_dim],
      loc_scale[:, latent_dim:]],axis=1)

  dist = tfp.layers.IndependentNormal(latent_dim)(combined_loc_scale)
  output = tf.keras.Model(inputs=za_input, outputs=dist)(za)
  predictor_net = tf.keras.Model(inputs=[obs_input,action_input],outputs=output)


  actor_net = rpc_utils.ActorNet(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        encoder=encoder_net,
        predictor=predictor_net,
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

  agent_checkpointer = common.Checkpointer(
    agent = tf_agent,
    ckpt_dir=agent_dir)

  agent_checkpointer.initialize_or_restore()


  return tf_agent,tf_env



@gin.configurable
def evaluate_prediction_model(
    trained_agent_dir,
    stacked_steps=5,
    env_name='HalfCheetah-v2',
    use_default_predictor=False
    ):

    
    tf_agent,tf_env = get_agent(trained_agent_dir)
        
    def evaluate_proto(tf_env, actor_net, prob_dropout, cutoff=10,rnn_model=False,
                   num_eval_episodes=10, stochastic_z=False):
        actor_net._input_tensor_spec = actor_net._z_spec  # pylint: disable=protected-access
        assert cutoff >= 1  # We must use the observation to initialize z at the first
        # time step.
        r_vec = []
        r_cutoff_vec = []
        t_vec = []
        x_vec = []
        x_cutoff_vec = []
        network_state = ()
        def _zero_array(spec):
            shape = (1,) + spec.shape
            return tf.zeros(shape, spec.dtype)
        network_state = tf.nest.map_structure(_zero_array, actor_net.state_spec)
        def _get_x(tf_env):
            if (hasattr(tf_env.envs[0], 'gym') and hasattr(tf_env.envs[0].gym, 'data') and hasattr(tf_env.envs[0].gym.data, 'qpos')):
                return tf_env.envs[0].gym.data.qpos.flatten()[0]
         
        @tf.function
        def _get_a(z, step_type, network_state):
            a, network_state = super(actor_net.__class__, actor_net).call(
                z, step_type=step_type, network_state=network_state, training=False)
            a = a.sample()
            return a, network_state

        for _ in range(10):
            ts = tf_env.reset()
            total_r = 0.0
            z = None
            last_ten_states = []
            last_ten_actions = []
            for t in range(1000):
                if t == cutoff:
                    x_cutoff_vec.append(_get_x(tf_env))
                    r_cutoff_vec.append(tf.identity(total_r))
                if t >= cutoff and np.random.random() < prob_dropout:
                    assert z is not None
                    # Generate Z by prediction
                    if rnn_model:
                        input_trajs = tf.stack(last_ten_states,axis=1)
                        input_actions = tf.stack(last_ten_actions,axis=1)
                        z = actor_net._predictor.layers[-1](tf.concat([input_trajs,input_actions],axis=2),training=False)
                        if stochastic_z:
                            z = z.sample()
                        else:
                            z = z.mean()
                    else:
                        z = actor_net._predictor.layers[-1](  # pylint: disable=protected-access
                            tf.concat([z, a], axis=1), training=False)
                        if stochastic_z:
                            z = z.sample()
                        else:
                            z = z.mean()
                        input_trajs = tf.stack(last_ten_states,axis=1)
                        input_actions = tf.stack(last_ten_actions,axis=1)
                        #_ = actor_net._predictor.layers[-1](tf.concat([input_trajs,input_actions],axis=2),training=False)
                else:
                    # Generate Z using the current observation
                    z = actor_net._z_encoder(ts.observation, training=False)  # pylint: disable=protected-access
                    print("Fucked it")
                a,network_state = _get_a(z, step_type=ts.step_type,network_state=network_state)
                ts = tf_env.step(a)
                total_r += ts.reward
                print(t,total_r)
                if ts.is_last():
                    break

                last_ten_states.append(z)
                last_ten_actions.append(a)

                if len(last_ten_actions) > 10:
                    last_ten_states.pop(0)
                    last_ten_actions.pop(0)
                #print(t,total_r)
            r_vec.append(total_r)
            print(total_r)
            x_vec.append(_get_x(tf_env))
        actor_net._input_tensor_spec = actor_net._s_spec  # pylint: disable=protected-access
        avg_r = tf.reduce_mean(r_vec)
        avg_t = tf.reduce_mean(t_vec)
        avg_x = tf.reduce_mean(x_vec)
        avg_cutoff_r = tf.reduce_mean(r_cutoff_vec)
        avg_cutoff_x = tf.reduce_mean(x_cutoff_vec)

        print('Return = %.3f', avg_r)
        print('Duration = %.3f', avg_t)
        print('Final X = %.3f', avg_x)
        print('Cutoff Return = %.3f', avg_cutoff_r)
        print('Cutoff X = %.3f', avg_cutoff_x)
        return avg_r, avg_t, avg_x, avg_cutoff_r, avg_cutoff_x        
      
        


    evaluate_proto(tf_env,tf_agent._actor_network,1.0,rnn_model=True,stochastic_z=False)
        
def main(_):
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  tf.config.set_visible_devices([], 'GPU')

  tf.compat.v1.enable_v2_behavior()
  #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  #logging.set_verbosity(logging.INFO)


  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  if 'xm_parameters' in FLAGS and FLAGS.xm_parameters:
    hparams = json.loads(FLAGS.xm_parameters)
    with gin.unlock_config():
      for (key, value) in hparams.items():
        print('Setting: %s = %s' % (key, value))
        gin.bind_parameter(key, value)

  trained_agent_dir = FLAGS.trained_agent_dir
  evaluate_prediction_model(trained_agent_dir)

if __name__ == '__main__':
  flags.mark_flag_as_required('trained_agent_dir')
  
  app.run(main)
