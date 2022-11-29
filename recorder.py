import os
import os.path
import logging

import numpy

import gridstorm.trace as trace
logger = logging.getLogger(__name__)

class Recorder:
    def __init__(self, finishers_only):
        self._only_keep_finishers = finishers_only

class VideoRecorder(Recorder):
    def __init__(self, renderer, only_keep_finishers):
        super().__init__(only_keep_finishers)
        self._paths = []
        self._path = None
        self._renderer = renderer

    def start_path(self):
        assert self._path is None
        self._path = trace.BeliefTrace()

    def end_path(self, finished):
        self._path.append_action(None)
        if not self._only_keep_finishers or finished:
            self._paths.append(self._path)
        self._path = None

    def record_state(self, state):
        self._path.append_state(state)

    def record_belief(self, belief):
        self._path.append_potential_states(belief)

    def record_selected_action(self, action):
        self._path.append_action(action)

    def record_available_actions(self, actions):
        self._path.append_available_actions(actions)

    def record_allowed_actions(self, actions):
        self._path.append_considered_actions(actions)

    def trim_from_end(self, length):
        for path in self._paths:
            path.trim_from_end(length)

    def save(self, path, prefix):
        for i, trace in enumerate(self._paths):
            mp4file = os.path.join(path,f"{prefix}-{i}.mp4")
            logger.info(f"Rendering {mp4file}")
            self._renderer.record(mp4file, trace)


class LoggingRecorder(Recorder):
    """
    A very simple general purpose recorder

    """
    def __init__(self, only_keep_finishers):
        super().__init__(only_keep_finishers)
        self._paths = []
        self._observed_paths = []
        self._path = None
        self._observed_path = None

    def start_path(self):
        assert self._path is None
        assert self._observed_path is None
        self._path = []
        self._observed_path = []

    def end_path(self, finished):
        if not self._only_keep_finishers or finished:
            self._paths.append(self._path)
            self._observed_paths.append(self._observed_path)
        self._path = None
        self._observed_path = None

    def record_state(self, state):
        self._path.append(f"{state}")

    def record_belief(self, belief):
        self._observed_path.append(f"{belief}")

    def record_selected_action(self, action):
        self._path.append(f"--act={action}-->")

    def record_available_actions(self, actions):
        pass

    def record_allowed_actions(self, actions):
        pass

    def save(self, path, prefix):
        for path in self._paths:
            print(" ".join(path))
        for observed_path in self._observed_paths:
            print(" ".join(observed_path))

class StatsRecorder(Recorder):
    def __init__(self, only_keep_finishers):
        super().__init__(only_keep_finishers)
        self._nr_allowed = None
        self._nr_available = None
        self._nr_allowed_paths = []
        self._nr_available_paths = []

    def start_path(self):
        assert len(self._nr_allowed_paths) == len(self._nr_available_paths)
        self._nr_allowed = []
        self._nr_available = []

    def end_path(self, finished):
        if not self._only_keep_finishers or finished:
            self._nr_allowed_paths.append(self._nr_allowed)
            self._nr_available_paths.append(self._nr_available)
        self._nr_allowed = None
        self._nr_available = None

    def record_state(self, state):
        pass

    def record_belief(self, belief):
        pass

    def record_selected_action(self, action):
        pass

    def record_available_actions(self, actions):
        self._nr_available.append(len(actions))

    def record_allowed_actions(self, actions):
        self._nr_allowed.append(len(actions))

    def save(self, filepath, prefix):
        def average(l):
            return sum(l) / len(l)

        avg_allowed = []
        avg_available = []
        avg_fraction = []
        for allowed_path, available_path in zip(self._nr_allowed_paths, self._nr_available_paths):
            #print(f"{average(allowed_path)} out of {average(available_path)}")
            avg_allowed.append(average(allowed_path))
            avg_available.append(average(available_path))
            avg_fraction.append(avg_allowed[-1] / avg_available[-1])
#        print(f"{average(avg_allowed)} out of {average(avg_available)}")
        with open(os.path.join(filepath, prefix + ".stats"), 'w') as file:
            file.write(f"Available avg: {average(avg_fraction)}\nAvailable avg stdev: {numpy.std(avg_fraction)}")
