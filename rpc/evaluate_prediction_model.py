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
  rb_dir,
  env_name='HalfCheetah-v2',
  latent_dim=10,
  predictor_num_layers=2,
  actor_fc_layers=(),
  clip_mean=30.0,
  clip_max_stddev=10.0,
  clip_min_stddev=0.1,
  critic_obs_fc_layers=None,
  critic_action_fc_layers=None,
  critic_joint_fc_layers=(256, 256),
  replay_buffer_capacity=10000,
  trained_predictor_dir=None,
  stacked_steps=-1):

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

  predictor_net = tf.keras.Model(inputs=(obs_input, action_input),
                                   outputs=output)

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

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=tf_agent.collect_data_spec,
      batch_size=tf_env.batch_size,
      max_length=replay_buffer_capacity)

  rb_checkpointer = common.Checkpointer(
    replay_buffer = replay_buffer,
    ckpt_dir=rb_dir)

  rb_checkpointer.initialize_or_restore()

  trained_predictor = None
  if trained_predictor_dir != None:
    z_input = tf.keras.layers.Input(latent_dim*stacked_steps)
    a_input = tf.keras.layers.Input(action_spec.shape[0]*stacked_steps)
    za = tf.concat([z_input,a_input],axis=1)
    loc_scale = tf.keras.Sequential(
        4 * [tf.keras.layers.Dense(512, activation='relu')]+[tf.keras.layers.Dense(latent_dim,activation=_custom_activation,kernel_initializer='zeros')])(za)
    combined_loc_scale = loc_scale + z_input[:,-latent_dim:]
    trained_predictor = tf.keras.Model(inputs=(z_input, a_input),
                                    outputs=combined_loc_scale)

  return tf_agent,replay_buffer,trained_predictor


def _filter_invalid_transition(trajectories,unusedarg1):
        return tf.reduce_all(~trajectories.is_boundary()[:-1])

@gin.configurable
def evaluate_prediction_model(
    trained_agent_dir,
    eval_rb_dir,
    trained_predictor_dir=None,
    stacked_steps=5,
    env_name='HalfCheetah-v2',
    use_default_predictor=False
    ):

    
    tf_agent,rb,trained_predictor = get_agent(trained_agent_dir,eval_rb_dir,trained_predictor_dir=trained_predictor_dir,stacked_steps=stacked_steps)

    predictor,encoder = tf_agent._actor_network._predictor,tf_agent._actor_network._z_encoder
    



    evaluation_loss = tf.keras.metrics.Mean('evaluation_loss', dtype=tf.float32)

    dataset = rb.as_dataset(sample_batch_size=128,num_steps=stacked_steps+5).filter(_filter_invalid_transition)
    dataset.prefetch(10)
    iterator = iter(dataset)


    
    def evaluate_proto(trajectories,predictor):
      """
      batch_squash = utils.BatchSquash(1)
      batch_squash_1 = utils.BatchSquash(2)
      obs = batch_squash.flatten(trajectories.observation[:,4:])
      obs_1 = batch_squash_1.flatten(trajectories.observation[:,4:])
      
      tf.debugging.assert_equal(obs[0],obs_1)
      tf.print(obs[0].shape,obs_1.shape)
      tf.print(obs[0],summarize=-1)
      tf.print(obs_1,summarize=-1)

      latent_encodings_1 = encoder(obs_1,training=False).mean()
      latent_encodings = encoder(obs,training=False).mean()

      tf.print(latent_encodings.shape,latent_encodings_1.shape)
      tf.print(latent_encodings,summarize=-1)
      tf.print(latent_encodings_1,summarize=-1)

      latent_encodings = batch_squash.unflatten(latent_encodings)
      latent_encodings_1 = batch_squash_1.unflatten(latent_encodings_1)

      

      tf.debugging.assert_equal(latent_encodings,latent_encodings_1)
      prev_encodings = latent_encodings[:,:stacked_steps]
      prev_encodings_1 = latent_encodings_1[:,:stacked_steps]

      #tf.debugging.assert_equal(prev_encodings,prev_encodings_1)

      input_trajs = tf.reshape(prev_encodings,[-1,prev_encodings.shape[1]*prev_encodings.shape[2]])
      input_trajs_1 = tf.reshape(prev_encodings_1,[-1,prev_encodings_1.shape[1]*prev_encodings_1.shape[2]])

      #tf.debugging.assert_equal(prev_encodings,prev_encodings_1)
      """

      batch_squash = utils.BatchSquash(2)

      obs = batch_squash.flatten(trajectories.observation)
      latent_encodings = encoder(obs,training=False).mean()
      latent_encodings = batch_squash.unflatten(latent_encodings)
      prev_encodings = latent_encodings[:,:stacked_steps]
      input_trajs = tf.reshape(prev_encodings,[-1,prev_encodings.shape[1]*prev_encodings.shape[2]])

      actions = trajectories.action[:,:stacked_steps]
      actions = tf.reshape(actions,[-1,actions.shape[1]*actions.shape[2]])
      
      predicted_encoding = predictor((input_trajs,actions))

      output_encoding = latent_encodings[:,-1]


      losses = []
      losses.append(tf.keras.losses.MSE(output_encoding, predicted_encoding))
      

      
      for i in range(1,1):
        input_trajs = tf.concat([input_trajs,predicted_encoding],axis=1)
        input_trajs = input_trajs[:,-10:]
        tf.debugging.assert_equal(input_trajs,predicted_encoding)
        actions = trajectories.action[:,i:stacked_steps+i]
        actions = tf.reshape(actions,[-1,actions.shape[1]*actions.shape[2]])
        predicted_encoding = predictor((input_trajs,actions))
        output_encoding = latent_encodings[:,stacked_steps+i]
        losses.append(tf.keras.losses.MSE(output_encoding, predicted_encoding))
      
        
      return losses


    
    results = []
    for i in range(25):
      np.save("eval_results_frames_"+str(stacked_steps)+".npy",results)
      print("At epoch",i,results)
      dataset = rb.as_dataset(sample_batch_size=128,num_steps=stacked_steps+1).filter(_filter_invalid_transition)
      dataset.prefetch(10)
      iterator = iter(dataset)
      evaluate = common.function(evaluate_proto)
      trained_predictor = tf.keras.models.load_model(trained_predictor_dir+"/model_epoch_"+str(i))

      i = 0
      for trajectories,_ in iterator: #Adjust length
        i += 128  
        losses = evaluate(trajectories,trained_predictor)
        evaluation_loss(losses[0])    
        if i > 10000:
          break  

      results.append(evaluation_loss.result())
      evaluation_loss.reset_states()
    print("Final results",results)
    np.save("test_fixed_run_"+str(stacked_steps)+".npy",results)

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
  eval_rb_dir = FLAGS.eval_rb_dir
  trained_predictor_dir = FLAGS.trained_predictor_dir
  evaluate_prediction_model(trained_agent_dir,eval_rb_dir,trained_predictor_dir=trained_predictor_dir)

if __name__ == '__main__':
  flags.mark_flag_as_required('trained_agent_dir')
  flags.mark_flag_as_required('eval_rb_dir')
  flags.mark_flag_as_required('trained_predictor_dir')
  
  app.run(main)
