import gym
from gym import error, spaces
import stormpy
import numpy as np

class ShieldToGymWrapper(gym.Env):
    def __init__(self,
                 model,
                 shield,
                 cfg
    ):
        self._model = model
        self._simulator = stormpy.simulator.create_simulator(model, seed=42)
        self._simulator.set_full_observability(True) # We want to access the full state space for visualisations.
        self._shield = shield
        action_spec_count = [self._model.get_nr_available_actions(i) for i in range(1, self._model.nr_states)]
        self.nr_actions = max(action_spec_count)
        self.action_space = spaces.Discrete(self.nr_actions)
        self.step_count = 0
        self.episode_count = 0
        self.cost_ind = list(self._model.reward_models.keys()).index('costs')
        self.gain_ind = list(self._model.reward_models.keys()).index('gains')
        self.first = True
        self.maxsteps = cfg['maxsteps']
        self.goal_value = cfg['goal_value']
        self.obs_type = cfg['obs_type']
        self.valuations = True
        if self.valuations:
            self.keywords = list(self.get_observation_keywords())
            self.keywords.remove('start')
            low = np.zeros((len(self.keywords),),dtype=int)
            high = np.zeros((len(self.keywords),),dtype=int)
            for s in range(self._model.nr_states):
                obs = self.get_observation_from_type(s,self.obs_type)
                obs_val = self.get_valuation(obs)
                low = np.minimum(low,obs_val)
                high = np.maximum(high,obs_val)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=int)
        elif self.obs_type == 'BELIEF_SUPPORT':
            self.observation_space = spaces.Box(
                low=np.zeros((self._model.nr_states,),dtype=int),
                high=np.zeros((self._model.nr_states,),dtype=int),
                dtype=int
            )


    def restart(self):
        state, rew, labels = self._simulator.restart()
        self._shield.reset()
        self.step_count = 0
        self.sink_flag = False
        self.episode_count += 1
        return self.observe()

    def reset(self):
        return self.restart()

    def is_done(self):
        return self._model.is_sink_state(self._simulator._engine.get_current_state()) or self.step_count==self.maxsteps

    def step(self,action):
        state, rew, labels = self._simulator.step(int(action))
        self._shield.track(action, self._simulator._report_observation())
        self.step_count += 1
        if (self.is_done() and 'traps' in labels):
            rew[self.cost_ind] += self.goal_value if self.goal_value > 100 else 0
        elif (self.is_done() and 'goal' in labels):
            rew[self.gain_ind] += self.goal_value
        elif (self.is_done()):
            rew[self.cost_ind] += 0
        return self.observe(), self.cost_fn(rew), self.is_done(), labels

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

    def get_valuation(self,val_in):
        if self.obs_type == 'STATE_LEVEL':
            return [json_to_int(self._model.state_valuations.get_json(val_in)[i]) for i in
             self.keywords]
        else:
            return [json_to_int(self._model.observation_valuations.get_json(val_in)[i])
                    for i in self.keywords]

    def get_observation_from_type(self,s,obs_type):
        if obs_type == 'STATE_LEVEL':
            return s
        else:
            return self._model.get_observation(s)


def json_to_int(i):
    try:
        return int(i)
    except:
        return 0 if i=='false' else 1

