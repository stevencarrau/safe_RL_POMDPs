# discrete_sac_agent.py
"""
A Soft Actor-Critic Agent.

Implements the discrete version of Soft Actor-Critic (SAC) algorithm based on
"Discrete and Continuous Action Representation for Practical RL in Video Games" by Olivier Delalleau, Maxim Peter, Eloi Alonso, Adrien Logut (2020).
Paper: https://montreal.ubisoft.com/en/discrete-and-continuous-action-representation-for-practical-reinforcement-learning-in-video-games/
"""
# Using Type Annotations.
from __future__ import absolute_import, division, print_function

import collections
from typing import Callable, Optional, Text

import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from six.moves import zip
from tf_agents.agents import data_converter, tf_agent
from tf_agents.networks import encoding_network, network, utils,categorical_projection_network,lstm_encoding_network,normal_projection_network
from tf_agents.policies import actor_policy, tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common, eager_utils, nest_utils, object_identity
from tf_agents.specs import tensor_spec

SacLossInfo = collections.namedtuple(
    'SacLossInfo', ('critic_loss', 'actor_loss', 'alpha_loss'))


@gin.configurable
class DiscreteSacAgent(tf_agent.TFAgent):
    """A SAC Agent that supports discrete action spaces."""

    def __init__(self,
                 time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec,
                 critic_network: network.Network,
                 actor_network: network.Network,
                 actor_optimizer: types.Optimizer,
                 critic_optimizer: types.Optimizer,
                 alpha_optimizer: types.Optimizer,
                 actor_loss_weight: types.Float = 1.0,
                 critic_loss_weight: types.Float = 0.5,
                 alpha_loss_weight: types.Float = 1.0,
                 actor_policy_ctor: Callable[
                     ..., tf_policy.TFPolicy] = actor_policy.ActorPolicy,
                 critic_network_2: Optional[network.Network] = None,
                 target_critic_network: Optional[network.Network] = None,
                 target_critic_network_2: Optional[network.Network] = None,
                 target_update_tau: types.Float = 1.0,
                 target_update_period: types.Int = 1,
                 td_errors_loss_fn: types.LossFn = tf.math.squared_difference,
                 gamma: types.Float = 1.0,
                 reward_scale_factor: types.Float = 1.0,
                 initial_log_alpha: types.Float = 0.0,
                 use_log_alpha_in_alpha_loss: bool = True,
                 target_entropy: Optional[types.Float] = None,
                 gradient_clipping: Optional[types.Float] = None,
                 debug_summaries: bool = False,
                 summarize_grads_and_vars: bool = False,
                 train_step_counter: Optional[tf.Variable] = None,
                 observation_and_action_constraint_splitter: Optional[types.Splitter] = None,
                 name: Optional[Text] = None):
        """Creates a SAC Agent.
        Args:
          time_step_spec: A `TimeStep` spec of the expected time_steps.
          action_spec: A nest of BoundedTensorSpec representing the actions.
          critic_network: A function critic_network((observations, actions)) that
            returns the q_values for each observation and action.
          actor_network: A function actor_network(observation, action_spec) that
            returns action distribution.
          actor_optimizer: The optimizer to use for the actor network.
          critic_optimizer: The default optimizer to use for the critic network.
          alpha_optimizer: The default optimizer to use for the alpha variable.
          actor_loss_weight: The weight on actor loss.
          critic_loss_weight: The weight on critic loss.
          alpha_loss_weight: The weight on alpha loss.
          actor_policy_ctor: The policy class to use.
          critic_network_2: (Optional.)  A `tf_agents.network.Network` to be used as
            the second critic network during Q learning.  The weights from
            `critic_network` are copied if this is not provided.
          target_critic_network: (Optional.)  A `tf_agents.network.Network` to be
            used as the target critic network during Q learning. Every
            `target_update_period` train steps, the weights from `critic_network`
            are copied (possibly withsmoothing via `target_update_tau`) to `
            target_critic_network`.  If `target_critic_network` is not provided, it
            is created by making a copy of `critic_network`, which initializes a new
            network with the same structure and its own layers and weights.
            Performing a `Network.copy` does not work when the network instance
            already has trainable parameters (e.g., has already been built, or when
            the network is sharing layers with another).  In these cases, it is up
            to you to build a copy having weights that are not shared with the
            original `critic_network`, so that this can be used as a target network.
            If you provide a `target_critic_network` that shares any weights with
            `critic_network`, a warning will be logged but no exception is thrown.
          target_critic_network_2: (Optional.) Similar network as
            target_critic_network but for the critic_network_2. See documentation
            for target_critic_network. Will only be used if 'critic_network_2' is
            also specified.
          target_update_tau: Factor for soft update of the target networks.
          target_update_period: Period for soft update of the target networks.
          td_errors_loss_fn:  A function for computing the elementwise TD errors
            loss.
          gamma: A discount factor for future rewards.
          reward_scale_factor: Multiplicative scale for the reward.
          initial_log_alpha: Initial value for log_alpha.
          use_log_alpha_in_alpha_loss: A boolean, whether using log_alpha or alpha
            in alpha loss. Certain implementations of SAC use log_alpha as log
            values are generally nicer to work with.
          target_entropy: The target average policy entropy, for updating alpha. The
            default value is negative of the total number of actions.
          gradient_clipping: Norm length to clip gradients.
          debug_summaries: A bool to gather debug summaries.
          summarize_grads_and_vars: If True, gradient and network variable summaries
            will be written during training.
          train_step_counter: An optional counter to increment every time the train
            op is run.  Defaults to the global_step.
          name: The name of this agent. All variables in this module will fall under
            that name. Defaults to the class name.
        """
        tf.Module.__init__(self, name=name)

        self._check_action_spec(action_spec)
        self._observation_and_action_constraint_splitter = (observation_and_action_constraint_splitter)
        net_observation_spec = time_step_spec.observation


        if observation_and_action_constraint_splitter:
            net_observation_spec, _ = observation_and_action_constraint_splitter(net_observation_spec)
        critic_spec = (net_observation_spec, action_spec)

        self._critic_network_1 = critic_network

        if critic_network_2 is not None:
            self._critic_network_2 = critic_network_2
        else:
            self._critic_network_2 = critic_network.copy(name='CriticNetwork2')
            # Do not use target_critic_network_2 if critic_network_2 is None.
            target_critic_network_2 = None

        # Wait until critic_network_2 has been copied from critic_network_1 before
        # creating variables on both.
        self._critic_network_1.create_variables(critic_spec)
        self._critic_network_2.create_variables(critic_spec)

        if target_critic_network:
            target_critic_network.create_variables(critic_spec)

        self._target_critic_network_1 = (
            common.maybe_copy_target_network_with_checks(
                self._critic_network_1,
                target_critic_network,
                input_spec=critic_spec,
                name='TargetCriticNetwork1'))

        if target_critic_network_2:
            target_critic_network_2.create_variables(critic_spec)
        self._target_critic_network_2 = (
            common.maybe_copy_target_network_with_checks(
                self._critic_network_2,
                target_critic_network_2,
                input_spec=critic_spec,
                name='TargetCriticNetwork2'))

        if actor_network:
            actor_network.create_variables(net_observation_spec)
        self._actor_network = actor_network

        policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=False,observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)

        self._train_policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=True,observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)

        self._log_alpha = common.create_variable(
            'initial_log_alpha',
            initial_value=initial_log_alpha,
            dtype=tf.float32,
            trainable=True)

        if target_entropy is None:
            target_entropy = self._get_default_target_entropy(action_spec)

        self._use_log_alpha_in_alpha_loss = use_log_alpha_in_alpha_loss
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._alpha_optimizer = alpha_optimizer
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._alpha_loss_weight = alpha_loss_weight
        self._td_errors_loss_fn = td_errors_loss_fn
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._target_entropy = target_entropy
        self._gradient_clipping = gradient_clipping
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._update_target = self._get_target_updater(
            tau=self._target_update_tau, period=self._target_update_period)

        train_sequence_length = 2 if not critic_network.state_spec else None

        super(DiscreteSacAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy=policy,
            collect_policy=policy,
            train_sequence_length=train_sequence_length,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
        )

        self._as_transition = data_converter.AsTransition(
            self.data_context, squeeze_time_dim=(train_sequence_length == 2))

    def _check_action_spec(self, action_spec):
        # The original SAC implementation would throw an error here if there were discrete actions,
        # but the Hybrid variation of SAC does support discrete actions.
        pass

    def _get_default_target_entropy(self, action_spec):
        # If target_entropy was not passed, set it to -dim(A)/2.0
        # Note that the original default entropy target is -dim(A) in the SAC paper.
        # However this formulation has also been used in practice by the original
        # authors and has in our experience been more stable for gym/mujoco.
        flat_action_spec = tf.nest.flatten(action_spec)
        target_entropy = -np.sum([
            np.product(single_spec.shape.as_list())
            for single_spec in flat_action_spec
        ]) / 2.0
        return target_entropy

    def _initialize(self):
        """Returns an op to initialize the agent.
        Copies weights from the Q networks to the target Q network.
        """
        common.soft_variables_update(
            self._critic_network_1.variables,
            self._target_critic_network_1.variables,
            tau=1.0)
        common.soft_variables_update(
            self._critic_network_2.variables,
            self._target_critic_network_2.variables,
            tau=1.0)

    def _train(self, experience, weights):
        """Returns a train op to update the agent's networks.
        This method trains with the provided batched experience.
        Args:
          experience: A time-stacked trajectory object.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
        Returns:
          A train_op.
        Raises:
          ValueError: If optimizers are None and no default value was provided to
            the constructor.
        """
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action

        trainable_critic_variables = list(object_identity.ObjectIdentitySet(
            self._critic_network_1.trainable_variables +
            self._critic_network_2.trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_critic_variables, ('No trainable critic variables to '
                                                'optimize.')
            tape.watch(trainable_critic_variables)
            critic_loss = self._critic_loss_weight * self.critic_loss(
                time_steps,
                actions,
                next_time_steps,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)

        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
        self._apply_gradients(critic_grads, trainable_critic_variables,
                              self._critic_optimizer)

        trainable_actor_variables = self._actor_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_actor_variables, ('No trainable actor variables to '
                                               'optimize.')
            tape.watch(trainable_actor_variables)
            actor_loss = self._actor_loss_weight * self.actor_loss(
                time_steps, weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
        self._apply_gradients(actor_grads, trainable_actor_variables,
                              self._actor_optimizer)

        alpha_variable = [self._log_alpha]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert alpha_variable, 'No alpha variable to optimize.'
            tape.watch(alpha_variable)
            alpha_loss = self._alpha_loss_weight * self.alpha_loss(
                time_steps, weights=weights)
        tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
        alpha_grads = tape.gradient(alpha_loss, alpha_variable)
        self._apply_gradients(alpha_grads, alpha_variable, self._alpha_optimizer)

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='critic_loss', data=critic_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='alpha_loss', data=alpha_loss, step=self.train_step_counter)

        self.train_step_counter.assign_add(1)
        self._update_target()

        total_loss = critic_loss + actor_loss + alpha_loss

        extra = SacLossInfo(
            critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss)

        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def _loss(self,
              experience: types.NestedTensor,
              weights: Optional[types.Tensor] = None):
        """Returns the loss of the provided experience.
        Args:
          experience: A time-stacked trajectory object.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
        Returns:
          A `LossInfo` containing the loss for the experience.
        """
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action
        critic_loss = self._critic_loss_weight * self.critic_loss(
            time_steps,
            actions,
            next_time_steps,
            td_errors_loss_fn=self._td_errors_loss_fn,
            gamma=self._gamma,
            reward_scale_factor=self._reward_scale_factor,
            weights=weights,
            training=False)
        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')

        actor_loss = self._actor_loss_weight * self.actor_loss(
            time_steps, weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')

        alpha_loss = self._alpha_loss_weight * self.alpha_loss(
            time_steps, weights=weights)
        tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='critic_loss', data=critic_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='alpha_loss', data=alpha_loss, step=self.train_step_counter)

        total_loss = critic_loss + actor_loss + alpha_loss

        extra = SacLossInfo(
            critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss)

        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def _apply_gradients(self, gradients, variables, optimizer):
        # list(...) is required for Python3.
        grads_and_vars = list(zip(gradients, variables))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                             self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)

        optimizer.apply_gradients(grads_and_vars)

    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.
        For each weight w_s in the original network, and its corresponding
        weight w_t in the target network, a soft update is:
        w_t = (1- tau) x w_t + tau x ws
        Args:
          tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
          period: Step interval at which the target network is updated.
        Returns:
          A callable that performs a soft update of the target network parameters.
        """
        with tf.name_scope('update_target'):
            def update():
                """Update target network."""
                critic_update_1 = common.soft_variables_update(
                    self._critic_network_1.variables,
                    self._target_critic_network_1.variables,
                    tau,
                    tau_non_trainable=1.0)

                critic_2_update_vars = common.deduped_network_variables(
                    self._critic_network_2, self._critic_network_1)

                target_critic_2_update_vars = common.deduped_network_variables(
                    self._target_critic_network_2, self._target_critic_network_1)

                critic_update_2 = common.soft_variables_update(
                    critic_2_update_vars,
                    target_critic_2_update_vars,
                    tau,
                    tau_non_trainable=1.0)

                return tf.group(critic_update_1, critic_update_2)

            return common.Periodically(update, period, 'update_targets')

    def _actions_and_log_probs(self, time_steps):
        """Get actions and corresponding log probabilities from policy."""
        # Get raw action distribution from policy, and initialize bijectors list.
        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        policy_state = self._train_policy.get_initial_state(batch_size)
        action_distribution = self._train_policy.distribution(
            time_steps, policy_state=policy_state).action

        # Sample actions and log_pis from transformed distribution.
        actions = tf.nest.map_structure(lambda d: d.sample(), action_distribution)
        log_pi = common.log_probability(action_distribution, actions,
                                        self.action_spec)

        return actions, log_pi

    def critic_loss(self,
                    time_steps: ts.TimeStep,
                    actions: types.Tensor,
                    next_time_steps: ts.TimeStep,
                    td_errors_loss_fn: types.LossFn,
                    gamma: types.Float = 1.0,
                    reward_scale_factor: types.Float = 1.0,
                    weights: Optional[types.Tensor] = None,
                    training: bool = False) -> types.Tensor:
        """Computes the critic loss for SAC training.
        Args:
          time_steps: A batch of timesteps.
          actions: A batch of actions.
          next_time_steps: A batch of next timesteps.
          td_errors_loss_fn: A function(td_targets, predictions) to compute
            elementwise (per-batch-entry) loss.
          gamma: Discount for future rewards.
          reward_scale_factor: Multiplicative factor to scale rewards.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
          training: Whether this loss is being used for training.
        Returns:
          critic_loss: A scalar critic loss.
        """
        with tf.name_scope('critic_loss'):
            nest_utils.assert_same_structure(actions, self.action_spec)
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)
            nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

            next_actions, next_log_pis = self._actions_and_log_probs(next_time_steps)
            target_input = (next_time_steps.observation['obs'], next_actions)
            target_q_values1, unused_network_state1 = self._target_critic_network_1(
                target_input, step_type=next_time_steps.step_type, training=False)
            target_q_values2, unused_network_state2 = self._target_critic_network_2(
                target_input, step_type=next_time_steps.step_type, training=False)
            target_q_values = (
                    tf.minimum(target_q_values1, target_q_values2) -
                    tf.exp(self._log_alpha) * next_log_pis)

            td_targets = tf.stop_gradient(
                reward_scale_factor * next_time_steps.reward +
                gamma * next_time_steps.discount * target_q_values)

            pred_input = (time_steps.observation['obs'], actions)
            pred_td_targets1, _ = self._critic_network_1(
                pred_input, step_type=time_steps.step_type, training=training)
            pred_td_targets2, _ = self._critic_network_2(
                pred_input, step_type=time_steps.step_type, training=training)
            critic_loss1 = td_errors_loss_fn(td_targets, pred_td_targets1)
            critic_loss2 = td_errors_loss_fn(td_targets, pred_td_targets2)
            critic_loss = critic_loss1 + critic_loss2

            if critic_loss.shape.rank > 1:
                # Sum over the time dimension.
                critic_loss = tf.reduce_sum(
                    critic_loss, axis=range(1, critic_loss.shape.rank))

            agg_loss = common.aggregate_losses(
                per_example_loss=critic_loss,
                sample_weight=weights,
                regularization_loss=(self._critic_network_1.losses +
                                     self._critic_network_2.losses))
            critic_loss = agg_loss.total_loss

            self._critic_loss_debug_summaries(td_targets, pred_td_targets1,
                                              pred_td_targets2)

            return critic_loss

    def actor_loss(self,
                   time_steps: ts.TimeStep,
                   weights: Optional[types.Tensor] = None) -> types.Tensor:
        """Computes the actor_loss for SAC training.
        Args:
          time_steps: A batch of timesteps.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
        Returns:
          actor_loss: A scalar actor loss.
        """
        with tf.name_scope('actor_loss'):
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)

            actions, log_pi = self._actions_and_log_probs(time_steps)
            target_input = (time_steps.observation['obs'], actions)
            target_q_values1, _ = self._critic_network_1(
                target_input, step_type=time_steps.step_type, training=False)
            target_q_values2, _ = self._critic_network_2(
                target_input, step_type=time_steps.step_type, training=False)
            target_q_values = tf.minimum(target_q_values1, target_q_values2)
            actor_loss = tf.exp(self._log_alpha) * log_pi - target_q_values
            if actor_loss.shape.rank > 1:
                # Sum over the time dimension.
                actor_loss = tf.reduce_sum(
                    actor_loss, axis=range(1, actor_loss.shape.rank))
            reg_loss = self._actor_network.losses if self._actor_network else None
            agg_loss = common.aggregate_losses(
                per_example_loss=actor_loss,
                sample_weight=weights,
                regularization_loss=reg_loss)
            actor_loss = agg_loss.total_loss
            self._actor_loss_debug_summaries(actor_loss, actions, log_pi,
                                             target_q_values, time_steps)

            return actor_loss

    def alpha_loss(self,
                   time_steps: ts.TimeStep,
                   weights: Optional[types.Tensor] = None) -> types.Tensor:
        """Computes the alpha_loss for EC-SAC training.
        Args:
          time_steps: A batch of timesteps.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
        Returns:
          alpha_loss: A scalar alpha loss.
        """
        with tf.name_scope('alpha_loss'):
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)

            unused_actions, log_pi = self._actions_and_log_probs(time_steps)
            entropy_diff = tf.stop_gradient(-log_pi - self._target_entropy)
            if self._use_log_alpha_in_alpha_loss:
                alpha_loss = (self._log_alpha * entropy_diff)
            else:
                alpha_loss = (tf.exp(self._log_alpha) * entropy_diff)

            if alpha_loss.shape.rank > 1:
                # Sum over the time dimension.
                alpha_loss = tf.reduce_sum(
                    alpha_loss, axis=range(1, alpha_loss.shape.rank))

            agg_loss = common.aggregate_losses(
                per_example_loss=alpha_loss, sample_weight=weights)
            alpha_loss = agg_loss.total_loss

            self._alpha_loss_debug_summaries(alpha_loss, entropy_diff)

            return alpha_loss

    def _critic_loss_debug_summaries(self, td_targets, pred_td_targets1,
                                     pred_td_targets2):
        if self._debug_summaries:
            td_errors1 = td_targets - pred_td_targets1
            td_errors2 = td_targets - pred_td_targets2
            td_errors = tf.concat([td_errors1, td_errors2], axis=0)
            common.generate_tensor_summaries('td_errors', td_errors,
                                             self.train_step_counter)
            common.generate_tensor_summaries('td_targets', td_targets,
                                             self.train_step_counter)
            common.generate_tensor_summaries('pred_td_targets1', pred_td_targets1,
                                             self.train_step_counter)
            common.generate_tensor_summaries('pred_td_targets2', pred_td_targets2,
                                             self.train_step_counter)

    def _actor_loss_debug_summaries(self, actor_loss, actions, log_pi,
                                    target_q_values, time_steps):
        if self._debug_summaries:
            common.generate_tensor_summaries('actor_loss', actor_loss,
                                             self.train_step_counter)
            try:
                for name, action in nest_utils.flatten_with_joined_paths(actions):
                    common.generate_tensor_summaries(name, action,
                                                     self.train_step_counter)
            except ValueError:
                pass  # Guard against internal SAC variants that do not directly
                # generate actions.

            common.generate_tensor_summaries('log_pi', log_pi,
                                             self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='entropy_avg',
                data=-tf.reduce_mean(input_tensor=log_pi),
                step=self.train_step_counter)
            common.generate_tensor_summaries('target_q_values', target_q_values,
                                             self.train_step_counter)
            batch_size = nest_utils.get_outer_shape(time_steps,
                                                    self._time_step_spec)[0]
            policy_state = self._train_policy.get_initial_state(batch_size)
            action_distribution = self._train_policy.distribution(
                time_steps, policy_state).action
            if isinstance(action_distribution, tfp.distributions.Normal):
                common.generate_tensor_summaries('act_mean', action_distribution.loc,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('act_stddev',
                                                 action_distribution.scale,
                                                 self.train_step_counter)
            elif isinstance(action_distribution, tfp.distributions.Categorical):
                common.generate_tensor_summaries('act_mode', action_distribution.mode(),
                                                 self.train_step_counter)
            try:
                for name, action_dist in nest_utils.flatten_with_joined_paths(
                        action_distribution):
                    common.generate_tensor_summaries('entropy_' + name,
                                                     action_dist.entropy(),
                                                     self.train_step_counter)
            except NotImplementedError:
                pass  # Some distributions do not have an analytic entropy.

    def _alpha_loss_debug_summaries(self, alpha_loss, entropy_diff):
        if self._debug_summaries:
            common.generate_tensor_summaries('alpha_loss', alpha_loss,
                                             self.train_step_counter)
            common.generate_tensor_summaries('entropy_diff', entropy_diff,
                                             self.train_step_counter)

            tf.compat.v2.summary.scalar(
                name='log_alpha', data=self._log_alpha, step=self.train_step_counter)


def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
  return categorical_projection_network.CategoricalProjectionNetwork(
      action_spec, logits_init_output_factor=logits_init_output_factor)


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1):
  std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      init_means_output_factor=init_means_output_factor,
      std_bias_initializer_value=std_bias_initializer_value)


class ActorDistributionRnnNetwork(network.DistributionNetwork):
  """Creates an actor producing either Normal or Categorical distribution.
  Note: By default, this network uses `NormalProjectionNetwork` for continuous
  projection which by default uses `tanh_squash_to_spec` to normalize its
  output. Due to the nature of the `tanh` function, values near the spec bounds
  cannot be returned.
  """

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               input_fc_layer_params=(200, 100),
               input_dropout_layer_params=None,
               lstm_size=None,
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               dtype=tf.float32,
               discrete_projection_net=_categorical_projection_net,
               continuous_projection_net=_normal_projection_net,
               rnn_construction_fn=None,
               rnn_construction_kwargs={},
               name='ActorDistributionRnnNetwork'):
    """Creates an instance of `ActorDistributionRnnNetwork`.
    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      input_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied before
        the LSTM cell.
      input_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent`, if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of input_fc_layer_params, or
        be None.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      dtype: The dtype to use by the convolution and fully connected layers.
      discrete_projection_net: Callable that generates a discrete projection
        network to be called with some hidden state and the outer_rank of the
        state.
      continuous_projection_net: Callable that generates a continuous projection
        network to be called with some hidden state and the outer_rank of the
        state.
      rnn_construction_fn: (Optional.) Alternate RNN construction function, e.g.
        tf.keras.layers.LSTM, tf.keras.layers.CuDNNLSTM. It is invalid to
        provide both rnn_construction_fn and lstm_size.
      rnn_construction_kwargs: (Optional.) Dictionary or arguments to pass to
        rnn_construction_fn.
        The RNN will be constructed via:
        ```
        rnn_layer = rnn_construction_fn(**rnn_construction_kwargs)
        ```
      name: A string representing name of the network.
    Raises:
      ValueError: If 'input_dropout_layer_params' is not None.
    """
    if input_dropout_layer_params:
      raise ValueError('Dropout layer is not supported.')

    lstm_encoder = lstm_encoding_network.LSTMEncodingNetwork(
        input_tensor_spec=input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=activation_fn,
        rnn_construction_fn=rnn_construction_fn,
        rnn_construction_kwargs=rnn_construction_kwargs,
        dtype=dtype,
        name=name)

    def map_proj(spec):
      if tensor_spec.is_discrete(spec):
        return discrete_projection_net(spec)
      else:
        return continuous_projection_net(spec)

    projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
    output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        projection_networks)

    super(ActorDistributionRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=lstm_encoder.state_spec,
        output_spec=output_spec,
        name=name)

    self._lstm_encoder = lstm_encoder
    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, observation, step_type, network_state=(), training=False,mask=None):
    state, network_state = self._lstm_encoder(
        observation, step_type=step_type, network_state=network_state,
        training=training)
    outer_rank = nest_utils.get_outer_rank(observation, self.input_tensor_spec)

    discrete_actions_distributions = tf.nest.map_structure(
        lambda proj_net, out_spec: proj_net(state, outer_rank, training=training,mask=mask)[0] if tensor_spec.is_discrete(out_spec) else None,
        self._projection_networks, self._output_tensor_spec)

    # Isolate the discrete action distributions. There may be none.
    discrete_actions_distributions_pruned = tf.nest.flatten(discrete_actions_distributions)
    discrete_actions_distributions_pruned = [action for action in discrete_actions_distributions_pruned if action is not None]

    if discrete_actions_distributions_pruned:
        if not training:
            # Turn discrete action logits into one-hots.
            discrete_actions = tf.nest.map_structure(
                lambda d_actions_distribution: tf.one_hot(tf.argmax(d_actions_distribution.logits, axis=1),
                                                          tf.shape(d_actions_distribution.logits)[-1],
                                                          axis=1,
                                                          dtype=d_actions_distribution.dtype) if d_actions_distribution else None,
                discrete_actions_distributions_pruned)
        else:
            # Use logits to train.
            discrete_actions = tf.nest.map_structure(
                lambda d_actions_distribution: tf.nn.softmax(d_actions_distribution.logits) if d_actions_distribution else None,
                discrete_actions_distributions_pruned)

        # Flatten and concatenate all discrete actions.
        discrete_actions = tf.nest.flatten(discrete_actions)
        discrete_actions = tf.nest.map_structure(lambda tensor: tf.reshape(tensor, [tensor.shape[0], -1]), discrete_actions)
        discrete_actions = tf.concat(discrete_actions, axis=-1)

        # Cast to the state's dtype.
        discrete_actions = tf.cast(discrete_actions, state.dtype)

        # Flatten state
        state = tf.reshape(state, [state.shape[0], -1])

        # Concatenate the discrete actions to the original state.
        state = tf.concat((state, discrete_actions), axis=-1, name='state_and_discrete_actions')

    # Isolate the continuous action distributions.
    continuous_actions_distributions = tf.nest.map_structure(
        lambda proj_net, out_spec: proj_net(state, outer_rank, training=training,mask=mask)[0] if tensor_spec.is_continuous(out_spec) else None,
        self._projection_networks, self._output_tensor_spec)

    # Now we have a list of discrete actions (if any) and a list of continuous actions.
    # Next, go through the output spec and pick the right action from these two disjoint lists.
    output_actions = tf.nest.map_structure(lambda d_action, c_action, out_spec: d_action if tensor_spec.is_discrete(out_spec) else c_action,
                                           discrete_actions_distributions, continuous_actions_distributions, self._output_tensor_spec)

    return output_actions, network_state