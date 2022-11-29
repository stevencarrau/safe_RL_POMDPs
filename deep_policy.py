import tensorflow as tf
import tf_agents
import functools
from tf_agents.agents import TFAgent
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import tf_policy
import numpy as np
from tf_agents.replay_buffers import episodic_replay_buffer
from itertools import chain
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.networks import sequential,actor_distribution_network,actor_distribution_rnn_network,value_rnn_network,value_network
from tf_agents.networks.mask_splitter_network import MaskSplitterNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.utils import common
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils
from reinforce import MaskedReinforceAgent
from sac import DiscreteSacAgent,ActorDistributionRnnNetwork
from ppo import PPOAgent


def dense_layer(num_units):
    return tf.keras.layers.Dense(num_units,activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'))

def obs_selector(args='DQN'):
    if args == 'DQN':
        return tf_agents.agents.dqn.dqn_agent.DqnAgent
    elif args == 'DDQN':
        return tf_agents.agents.dqn.dqn_agent.DdqnAgent
    elif args == 'PPO':
        return PPOAgent
    elif args == 'DDPG':
        return tf_agents.agents.ddpg.ddpg_agent.DdpgAgent
    elif args == 'REINFORCE':
        return MaskedReinforceAgent
    elif args == 'TD3':
        return tf_agents.agents.td3.td3_agent.Td3Agent
    elif args == 'SAC':
        return DiscreteSacAgent

class DeepAgent():

    def __init__(self,env,alpha,agent_arg='PPO',layer_size=None):
        self.alpha = alpha
        self.learning_method(env,alpha,agent_arg,layer_size)

    def observation_and_action_constraint_splitter(self,observation):
        return observation['obs'], observation['mask']

    def create_actor_network(self, env, actor_fc_layers, distribution_out=True):
        if distribution_out:
            actor_net = actor_distribution_network.ActorDistributionNetwork(env.obs_spec['obs'],env.act_spec,fc_layer_params=actor_fc_layers)
        else:
            actor_net = ActorNetwork(env.obs_spec['obs'],env.act_spec,fc_layer_params=actor_fc_layers)
        return actor_net #MaskSplitterNetwork(self.observation_and_action_constraint_splitter,actor_net,passthrough_mask=True)


    def create_value_network(self,env,value_net_fc_layers,distribution_out=True):
        if distribution_out:
            value_net = value_network.ValueNetwork(env.obs_spec['obs'],fc_layer_params=value_net_fc_layers)
        else:
            value_net = value_network.ValueNetwork(env.obs_spec['obs'],fc_layer_params=value_net_fc_layers)
        return value_net #MaskSplitterNetwork(self.observation_and_action_constraint_splitter,value_net,passthrough_mask=True)

    # def create_critic_network(self,env,obs_fc_layer_units,action_fc_layer_units,joint_fc_layer_units):
    #     critic_net = DiscreteSacCriticNetwork((env.obs_spec['obs'],env.act_spec),observation_fc_layer_params=obs_fc_layer_units,action_fc_layer_params=action_fc_layer_units,joint_fc_layer_params=joint_fc_layer_units)
    #     return critic_net # MaskSplitterNetwork(self.observation_and_action_constraint_splitter,critic_net,passthrough_mask=True)

    def learning_method(self,env,alpha,agent_arg,layer_size=None):
        train_step_counter = tf.Variable(0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
        if agent_arg == 'DQN':
            if layer_size:
                layer_params = (layer_size,)
            else:
                layer_params = (100,)
            self.fc_layer_params = layer_params
            dense_layers = [dense_layer(num_units) for num_units in self.fc_layer_params]
            q_values_layer = tf.keras.layers.Dense(env.nr_actions, activation=None,
                                                   kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03,
                                                                                                          maxval=0.03),
                                                   bias_initializer=tf.keras.initializers.Constant(-0.2))
            q_net = tf_agents.networks.sequential.Sequential(dense_layers + [q_values_layer])
            self.agent = obs_selector(agent_arg)(env.time_step_spec, env.act_spec, q_network=q_net, optimizer=optimizer,
                             td_errors_loss_fn=tf_agents.utils.common.element_wise_squared_loss,
                             train_step_counter=train_step_counter,
                             observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter)
        elif agent_arg == 'DDQN':
            if layer_size:
                layer_params = (layer_size,)
            else:
                layer_params = (100,)
            self.fc_layer_params = layer_params
            dense_layers = [dense_layer(num_units) for num_units in self.fc_layer_params]
            q_values_layer = tf.keras.layers.Dense(env.nr_actions, activation=None,
                                                   kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03,
                                                                                                          maxval=0.03),
                                                   bias_initializer=tf.keras.initializers.Constant(-0.2))
            q_net = tf_agents.networks.sequential.Sequential(dense_layers + [q_values_layer])
            self.agent = obs_selector(agent_arg)(env.time_step_spec, env.act_spec, q_network=q_net, optimizer=optimizer,
                                                 td_errors_loss_fn=tf_agents.utils.common.element_wise_squared_loss,
                                                 train_step_counter=train_step_counter,
                                                 observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter)
        elif agent_arg == 'PPO':
            if layer_size:
                actor_fc_layers=(layer_size, 100)
                value_fc_layers=(layer_size, 100)
            else:
                actor_fc_layers=(200, 100)
                value_fc_layers=(200, 100)
            self.fc_layer_params = actor_fc_layers
            use_rnns=False
            lstm_size=(20,)
            actor_net = self.create_actor_network(env,actor_fc_layers,True)
            value_net = self.create_value_network(env,value_fc_layers,True)
            self.agent = obs_selector(agent_arg)(
                env.time_step_spec,
                env.act_spec,
                optimizer,
                actor_net=actor_net,
                value_net=value_net,
                entropy_regularization=0.0,
                importance_ratio_clipping=0.2,
                normalize_observations=False,
                normalize_rewards=False,
                use_gae=True,
                train_step_counter=train_step_counter,observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter)
        elif agent_arg == 'DDPG':
            actor_fc_layers = (400, 300)
            critic_obs_fc_layers = (400,)
            critic_action_fc_layers = None
            critic_joint_fc_layers = (300,)
            ou_stddev = 0.2
            ou_damping = 0.15
            actor_learning_rate = 1e-4
            critic_learning_rate = 1e-3
            td_errors_loss_fn = tf.compat.v1.losses.huber_loss
            actor_net = self.create_actor_network(env,actor_fc_layers,False)
            critic_net = self.create_critic_network(env,critic_obs_fc_layers,critic_action_fc_layers,critic_joint_fc_layers)
            self.agent = obs_selector(agent_arg)(
                env.time_step_spec,
                env.act_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=critic_learning_rate),
                ou_stddev=ou_stddev,
                ou_damping=ou_damping,
                target_update_tau=0.05,
                target_update_period=5,
                dqda_clipping=None,
                td_errors_loss_fn=td_errors_loss_fn,
                train_step_counter=train_step_counter)
        elif agent_arg == 'REINFORCE':
            if layer_size:
                actor_fc_layers = (layer_size,)
                value_net_fc_layers = (layer_size,)
            else:
                actor_fc_layers = (100,)
                value_net_fc_layers = (100,)
            learning_rate=alpha
            value_estimation_loss_coef = 0.2
            actor_net = self.create_actor_network(env,actor_fc_layers,True)
            value_net = self.create_value_network(env,value_net_fc_layers,True)
            self.agent = obs_selector(agent_arg)(
                env.time_step_spec,
                env.act_spec,
                actor_network=actor_net,
                value_network=value_net,
                value_estimation_loss_coef=value_estimation_loss_coef,
                gamma=1.0,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
                train_step_counter=train_step_counter,
                observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter)

        elif agent_arg == 'TD3':
            actor_fc_layers = (400, 300)
            critic_obs_fc_layers = (400,)
            critic_action_fc_layers = None
            critic_joint_fc_layers = (300,)
            exploration_noise_std = 0.1
            target_update_tau = 0.05
            actor_update_period = 2
            actor_learning_rate = alpha
            critic_learning_rate = alpha*10
            target_update_period = 5
            actor_net = self.create_actor_network(env,actor_fc_layers,False)
            critic_net = self.create_critic_network(env,critic_obs_fc_layers,critic_action_fc_layers,critic_joint_fc_layers)
            self.agent = obs_selector(agent_arg)(
                env.time_step_spec,
                env.act_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=critic_learning_rate),
                exploration_noise_std=exploration_noise_std,
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                actor_update_period=actor_update_period,
                train_step_counter=train_step_counter,
            )
        elif agent_arg == 'SAC':
            actor_fc_layers = (400, 300)
            actor_output_fc_layers = (100,)
            actor_lstm_size = (40,)
            critic_obs_fc_layers = None
            critic_action_fc_layers = None
            critic_joint_fc_layers = (300,)
            critic_output_fc_layers = (100,)
            critic_lstm_size = (40,)
            target_update_tau = 0.05
            target_update_period = 5
            actor_learning_rate = alpha
            critic_learning_rate = alpha
            alpha_learning_rate = alpha
            td_errors_loss_fn = tf.math.squared_difference
            reward_scale_factor = 0.1
            actor_net = ActorDistributionRnnNetwork(env.obs_spec['obs'],env.act_spec,input_fc_layer_params=actor_fc_layers,lstm_size=actor_lstm_size,output_fc_layer_params=actor_output_fc_layers,continuous_projection_net=tanh_normal_projection_network
        .TanhNormalProjectionNetwork)
            critic_net = critic_rnn_network.CriticRnnNetwork((env.obs_spec['obs'], env.act_spec),observation_fc_layer_params=critic_obs_fc_layers,action_fc_layer_params=critic_action_fc_layers,joint_fc_layer_params=critic_joint_fc_layers,lstm_size=critic_lstm_size,output_fc_layer_params=critic_output_fc_layers,kernel_initializer='glorot_uniform',last_kernel_initializer='glorot_uniform')
            self.agent = obs_selector(agent_arg)(
                env.time_step_spec,
                env.act_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=critic_learning_rate),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=alpha_learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=td_errors_loss_fn,
                gamma=1.0,
                reward_scale_factor=reward_scale_factor,
                gradient_clipping=None,
                train_step_counter=train_step_counter,observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter)


class ActorNetwork(network.Network):

  def __init__(self,
               observation_spec,
               action_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(75, 40),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               enable_last_layer_zero_initializer=False,
               name='ActorNetwork'):
    super(ActorNetwork, self).__init__(
        input_tensor_spec=observation_spec, state_spec=(), name=name)

    self._action_spec = action_spec
    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]
    if self._single_action_spec.dtype not in [tf.int32, tf.int64]:
      raise ValueError('Only int actions are supported by this network.')

    kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1. / 3., mode='fan_in', distribution='uniform')
    self._encoder = encoding_network.EncodingNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)

    initializer = tf.keras.initializers.RandomUniform(
        minval=-0.003, maxval=0.003)

    self._action_projection_layer = tf.keras.layers.Dense(
        flat_action_spec[0].shape.num_elements(),
        activation=tf.keras.activations.tanh,
        kernel_initializer=initializer,
        name='action')

  def call(self, observations, step_type=(), network_state=()):
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    # We use batch_squash here in case the observations have a time sequence
    # compoment.
    batch_squash = utils.BatchSquash(outer_rank)
    observations = tf.nest.map_structure(batch_squash.flatten, observations)

    state, network_state = self._encoder(
        observations, step_type=step_type, network_state=network_state)
    actions = self._action_projection_layer(state)
    actions = common.scale_to_spec(actions, self._single_action_spec)
    actions = batch_squash.unflatten(actions)
    return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state

class ReplayMemory:
    def __init__(self, config):
        self.config = config
        self.actions = np.empty((self.config['mem_size']-1), dtype=np.int32)
        self.rewards = np.empty((self.config['mem_size']-1), dtype=np.int32)
        self.observations = np.empty((self.config['mem_size'], self.config['obs_dims']), dtype=np.int32)
        self.count = 0
        self.current = 0

    def reset(self,obs):
        self.actions = np.empty((self.config['mem_size'] - 1), dtype=np.int32)
        self.rewards = np.empty((self.config['mem_size'] - 1), dtype=np.int32)
        self.observations = np.empty((self.config['mem_size'], self.config['obs_dims']), dtype=np.int32)
        self.count = 0
        self.current = 0
        self.initial_add(obs)

    def initial_add(self,obs):
        self.observations[-1,:] = obs

    def add(self, action, reward,next_obs):
        if self.config['mem_size']>1:
            self.count = max(self.count, self.current + 1)
            for i in range(self.count - 1):
                self.actions[i] = self.actions[i]
                self.rewards[i] = self.rewards[i]
                self.observations[i] = self.observations[i + 1]
            self.observations[-1,:] = next_obs
            self.actions[-1] = action
            self.rewards[-1] = reward
            self.current = (self.current + 1) % self.config['mem_size']
        else:
            self.observations[-1, :] = next_obs

    def getObs(self):
        return self.observations.reshape(-1)

    def getSeq(self):
        seq_out = list(chain.from_iterable(zip(self.observations[:-1],self.actions)))
        seq_out.append(self.observations[-1])
        return np.array(seq_out).reshape(-1)

    def getRewSeq(self):
        seq_out = list(chain.from_iterable(zip(self.observations[:-1], self.actions,self.rewards)))
        seq_out.append(self.observations[-1])
        return np.array(seq_out).reshape(-1)
    # def __call__(self,policy, obs, allowed_actions=None):
    #     act_logits,_ = policy.distribution(obs)
    #     if allowed_actions:
    #         mask = np.zeros_like(act_logits.numpy(), dtype=bool)
    #         for i in allowed_actions:
    #             mask[0, i] = True
    #         new_logits = tf_agents.distributions.masked.MaskedCategorical(logits=act_logits, mask=mask)
    #         action = tf.random.categorical(logits=new_logits.logits, num_samples=1, dtype=None, seed=None,
    #                                            name=None).numpy()[0, 0]
    #     else:
    #         action = tf.random.categorical(logits=act_logits, num_samples=1, dtype=None, seed=None,
    #                                            name=None).numpy()[0, 0]
    #     return tf_agents.trajectories.policy_step.PolicyStep(action=tf.constant([action]))


