import random

import numpy as np
import stormpy as sp
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import stormpy.pomdp




import logging
logger = logging.getLogger(__name__)



class Tracker:
    """
    Wraps the belief support tracker for our purposes
    """
    def __init__(self, model, shield):
        self._model = model
        self._tracker = stormpy.pomdp.BeliefSupportTrackerDouble(model)
        self._shield = shield

    def track(self, action, observation):
        logger.debug(f"Track action={action}, observation={observation}")
        self._tracker.track(action, observation)

    def monitor(self):
        result = self._shield.query_current_belief(self._tracker.get_current_belief_support())
        logger.debug("Current belief is {}".format("safe" if result else "not safe"))
        return result

    def shielded_actions(self, action_indices):
        safe_action_indices = []
        for a in action_indices:
            if self._shield.query_action(self._tracker.get_current_belief_support(), a):
                safe_action_indices.append(a)
        return safe_action_indices

    def list_support(self):
        return [s for s in self._tracker.get_current_belief_support()]

    def reset(self):
        self._tracker = stormpy.pomdp.BeliefSupportTrackerDouble(self._model)

class SimulationExecutor:
    """
    Base class that wraps and extends the stormpy simulator for shielding.
    """
    def __init__(self, model, shield):
        self._model = model
        self._simulator = stormpy.simulator.create_simulator(model, seed=10)
        self._simulator.set_full_observability(True) # We want to access the full state space for visualisations.
        self._shield = shield

    def simulate(self, recorder, nr_good_runs = 1, total_nr_runs = 5, maxsteps=30):
        result = []
        good_runs = 0
        #TODO what if we are not in a safe state.
        for m in range(total_nr_runs):
            finished = False
            state = self._simulator.restart()
            logger.info("Start new episode.")
            self._shield.reset()
            recorder.start_path()
            recorder.record_state(state)
            recorder.record_belief(self._shield.list_support())
            for n in range(maxsteps):
                actions = self._simulator.available_actions()
                safe_actions = self._shield.shielded_actions(range(len(actions)))
                logger.debug(f"Number of actions: {actions}. Safe action indices: {safe_actions}")
                if len(safe_actions) == 0:
                    select_action = random.randint(0, len(actions) - 1)
                    action = actions[select_action]
                else:
                    select_action = random.randint(0, len(safe_actions) - 1)
                    action = safe_actions[select_action]
                logger.debug(f"Select action: {action}")
                state, rew,labels = self._simulator.step(action)
                self._shield.track(action, self._model.get_observation(state))
                assert state in self._shield.list_support()
                logger.debug(f"Now in state {state}. Belief: {self._shield.list_support()}. Safe: {self._shield.monitor()}")

                recorder.record_available_actions(actions)
                recorder.record_allowed_actions(safe_actions)
                recorder.record_selected_action(action)
                recorder.record_state(state)
                recorder.record_belief(self._shield.list_support())

                if self._simulator.is_done():
                    logger.info(f"Done after {n} steps!")
                    finished = True
                    good_runs += 1
                    break
            actions = self._simulator.available_actions()
            safe_actions = self._shield.shielded_actions(range(len(actions)))
            print(safe_actions)

            recorder.record_available_actions(actions)
            recorder.record_allowed_actions(safe_actions)

            recorder.end_path(finished)
            result.append(self._simulator.is_done())
            if good_runs == nr_good_runs:
                break
        return result

def json_to_int(i):
    try:
        return int(i)
    except:
        return 0 if i=='false' else 1

class SimulationWrapper(SimulationExecutor):
    def __init__(self, model, shield,goal_value):
        super(SimulationWrapper, self).__init__(model,shield)
        self.goal_value = goal_value
        action_keywords = set()
        for s_i in range(self._model.nr_states):
            n_act = self._model.get_nr_available_actions(s_i)
            for a_i in range(n_act):
                action_keywords = action_keywords.union(
                    self._model.choice_labeling.get_labels_of_choice(self._model.get_choice_index(s_i, a_i)))
        # action_spec_count = [self._model.get_nr_available_actions(i) for i in range(1,self._model.nr_states)]
        self.action_indices = dict([[j, i] for i, j in enumerate(action_keywords)])
        self.act_keywords = dict([[self.action_indices[i], i] for i in self.action_indices])

    def simulate(self, recorder, nr_good_runs = 1, total_nr_runs = 5, maxsteps=30):
        result = []
        good_runs = 0
        for m in range(total_nr_runs):
            finished = False
            state = self._simulator.restart()
            logger.info("Start new episode.")
            self._shield.reset()
            recorder.start_path()
            recorder.record_state(state)
            recorder.record_belief(self._shield.list_support())
            for n in range(maxsteps):
                actions = self._simulator.available_actions()
                safe_actions = self._shield.shielded_actions(range(len(actions)))
                logger.debug(f"Number of actions: {actions}. Safe action indices: {safe_actions}")
                if len(safe_actions) == 0:
                    select_action = random.randint(0, len(actions) - 1)
                    action = actions[select_action]
                else:
                    select_action = random.randint(0, len(safe_actions) - 1)
                    action = safe_actions[select_action]
                logger.debug(f"Select action: {action}")
                state, rew,labels = self._simulator.step(action)
                self._shield.track(action, self._model.get_observation(state))
                assert state in self._shield.list_support()
                logger.debug(f"Now in state {state}. Belief: {self._shield.list_support()}. Safe: {self._shield.monitor()}")

                recorder.record_available_actions(actions)
                recorder.record_allowed_actions(safe_actions)
                recorder.record_selected_action(action)
                recorder.record_state(state)
                recorder.record_belief(self._shield.list_support())

                if self._simulator.is_done():
                    logger.info(f"Done after {n} steps!")
                    finished = True
                    good_runs += 1
                    break
            actions = self._simulator.available_actions()
            safe_actions = self._shield.shielded_actions(range(len(actions)))
            print(safe_actions)

            recorder.record_available_actions(actions)
            recorder.record_allowed_actions(safe_actions)

            recorder.end_path(finished)
            result.append(self._simulator.is_done())
            if good_runs == nr_good_runs:
                break
        return result

    def restart(self):
        self._simulator.restart()
        self._shield.reset()
        return

    def observe(self):
        if self.valuations:
            return np.array(self.get_observation_valuation(),dtype=int)
        else:
            if self.obs_type == 'STATE_LEVEL':
                return [self._simulator._report_state()]
            elif self.obs_type == 'BELIEF_SUPPORT':
                support = np.zeros((self._model.nr_states,),dtype=int)
                for i in self._shield.list_support():
                    support[i] = 1
                return support.tolist()
            else:
                return [self._simulator._report_observation()]

    def cost_fn(self,rew_in):
        if len(rew_in)>1:
            return rew_in[self.gain_ind]-rew_in[self.cost_ind]
        else:
            return -rew_in[0]

    def get_observation_valuation(self):
        if self.obs_type == 'STATE_LEVEL':
            return [json_to_int(self._model.state_valuations.get_json(self._simulator._report_state())[i]) for i in self.keywords]
        else:
            return [json_to_int(self._model.observation_valuations.get_json(self._simulator._report_observation())[i]) for i in self.keywords]

    def get_observation_keywords(self):
        if self.obs_type == 'STATE_LEVEL':
            keywords = set([i[1:-2] for i in str(self._model.state_valuations.get_json(0)).split()[1:-1:2]])
        else:
            keywords = set([i[1:-2] for i in str(self._model.observation_valuations.get_json(0)).split()[1:-1:2]])
        return keywords

    def get_choice_labels(self):
        return [self._model.choice_labeling.get_labels_of_choice(self._model.get_choice_index(self._simulator._report_state(),a_i)).pop() for a_i in range(self._simulator.nr_available_actions())]

    def apply_action(self,act_in):
        act_keyword = self.act_keywords[int(act_in)]
        choice_list = self.get_choice_labels()
        return choice_list.index(act_keyword)
