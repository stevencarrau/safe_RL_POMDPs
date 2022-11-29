import argparse
import copy

import numpy as np
import stormpy as sp
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import stormpy.pomdp
import random
import os.path
import inspect

# from rlshield.noshield import NoShield
# from rlshield.recorder import LoggingRecorder, VideoRecorder, StatsRecorder
# from rlshield.model_simulator import SimulationExecutor, Tracker
from noshield import NoShield
from recorder import LoggingRecorder, VideoRecorder, StatsRecorder
from model_simulator import Tracker
from model_simulator import SimulationExecutor,SimulationWrapper
from rl_simulator import TF_Environment,solicit_input,compute_avg_return
import matplotlib.pyplot as plt



from gridstorm.plotter import Plotter


import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
experiment_names = [
    "avoid",
    "refuel",
    'obstacle',
    "intercept",
    'evade',
    'rocks'
]

def compute_winning_region(model, formula, initial=True):
    options = sp.pomdp.IterativeQualitativeSearchOptions()
    model = sp.pomdp.prepare_pomdp_for_qualitative_search_Double(model, formula)
    solver = sp.pomdp.create_iterative_qualitative_search_solver_Double(model, formula, options)
    logger.info("compute winning region...")
    if initial:
        solver.compute_winning_policy_for_initial_states(100)
    else:
        solver.compute_winning_region(100)
    logger.info("done.")
    return solver.last_winning_region

def construct_otf_shield(model, winning_region):
    return sp.pomdp.BeliefSupportWinningRegionQueryInterfaceDouble(model, winning_region)

def build_model(input):
    prism_program = sp.parse_prism_program(input.path)
    prop = sp.parse_properties_for_prism_program(input.properties[0],prism_program)[0]
    prism_program, props = stormpy.preprocess_symbolic_input(prism_program, [prop], input.constants)
    prop = props[0]
    prism_program = prism_program.as_prism_program()
    raw_formula = prop.raw_formula
    logger.info("Construct POMDP representation...")
    model = build_pomdp(prism_program, raw_formula)
    model = sp.pomdp.make_canonic(model)
    return model,prism_program,raw_formula

def build_pomdp(program, formula):
    options = stormpy.BuilderOptions([formula])
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()
    options.set_build_all_reward_models()
    options.set_build_observation_valuations()
    logger.debug("Start building the POMDP")
    return sp.build_sparse_model_with_options(program, options)



class ManualInput:
    def __init__(self, path, prop, constants):
        self.path = path
        self.properties = [prop]
        self.constants = constants
def main(cfg=None):
    if cfg:
        parser = argparse.ArgumentParser(description='The shielded POMDP simulator.')
        parser.add_argument('--prism', help="Specify model from prism file",default=False)
        parser.add_argument('--logfile', help="File to log to", default="rendering.log")
        parser.add_argument('--prop', help='Specify property string directly')
        parser.add_argument('--load-winning-region', '-wr', help="Load a winning region")
        parser.add_argument('--nr-finisher-runs', '-N', type=int, default=1)
        parser.add_argument('--video-path', help="Path for the video")
        parser.add_argument('--stats-path', help="Path for recording stats")
        parser.add_argument('--finishers-only', action='store_true')
        parser.add_argument('--seed', help="Seed for randomised movements", default=3)
        parser.add_argument('--title', help="Title for video")
        args = parser.parse_args()
        for c_i in cfg:
            args.__setattr__(c_i, cfg[c_i])
    else:
        parser = argparse.ArgumentParser(description='The shielded POMDP simulator.')
        model_group = parser.add_mutually_exclusive_group(required=True)
        model_group.add_argument('--grid-model', '-m', help=f'Model from the gridworld-by-storm visualisation set, choose from {str(experiment_names)}')
        model_group.add_argument('--prism', help="Specify model from prism file")
        parser.add_argument('--prop', help='Specify property string directly')
        parser.add_argument('--constants', '-c', help="Constants to select the instance of the model", default="")
        parser.add_argument('--load-winning-region', '-wr', help="Load a winning region")
        parser.add_argument('--maxsteps', '-s', help="Maximal number of steps", type=int, default=100)
        #parser.add_argument('--maxrendering', '-r', help='Maximal length of a rendering', type=int, default=100)
        parser.add_argument('--max-runs', '-NN', help="Number of runs", type=int, default=5000)
        parser.add_argument('--nr-finisher-runs', '-N', type=int, default=1)
        parser.add_argument('--video-path', help="Path for the video")
        parser.add_argument('--stats-path', help="Path for recording stats")
        parser.add_argument('--finishers-only', action='store_true')
        parser.add_argument('--seed', help="Seed for randomised movements", default=3)
        parser.add_argument('--title', help="Title for video")
        parser.add_argument('--logfile', help="File to log to", default="rendering.log")
        parser.add_argument('--noshield', help="Simulate without a shield", action='store_true')
        parser.add_argument('--obs_level',default='BELIEF_SUPPORT')
        parser.add_argument('--valuations',default=False)
        parser.add_argument('--learning_method',default='PPO')
        parser.add_argument('--eval-interval', type=int,default=100)
        parser.add_argument('--eval-episodes', type=int,default=5)
        parser.add_argument('--goal-value', type=int, default=10)

        args = parser.parse_args()
    logging.basicConfig(filename=f'{args.logfile}', filemode='w', level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    if args.prism and not args.prop:
        raise RuntimeError("Prism models require setting the property via --prop")
    if args.grid_model and args.prop:
        raise RuntimeError("Properties cannot be set manually when using the gridstorm models")

    if args.video_path is not None and not os.path.isdir(args.video_path):
        raise RuntimeError(f"Video path {args.video_path} not known!")
    if args.stats_path is not None and not os.path.isdir(args.stats_path):
        raise RuntimeError(f"Stats path {args.stats_path} not known!")

    if args.video_path and not args.grid_model:
        raise RuntimeError("Rendering is only supported for gridstorm models!")

    random.seed(args.seed)
    if args.grid_model:
        logger.info("Look up problem definition....")
        if hasattr(args,'full_obs'):
            if args.full_obs:
                import gridfull.models as models
                if hasattr(args,'nonsparse'):
                    if args.nonsparse:
                        import gridfullsparse.models as models_learn
                    else:
                        import gridfull.models as models_learn
                else:
                    import gridfull.models as models_learn
            else:
                import gridstorm.models as models
                if hasattr(args,'nonsparse'):
                    if args.nonsparse:
                        import gridsparse.models as models_learn
                    else:
                        import gridstorm.models as models_learn
                else:
                    import gridstorm.models as models_learn
        else:
            import gridstorm.models as models
            if hasattr(args, 'nonsparse'):
                if args.nonsparse:
                    import gridsparse.models as models_learn
                else:
                    import gridstorm.models as models_learn
            else:
                import gridstorm.models as models_learn
        experiment_to_grid_model_names = {"avoid": models_learn.surveillance, "refuel": models_learn.refuel,
                                          'obstacle': models_learn.obstacle, "intercept": models_learn.intercept,
                                          'evade': models_learn.evade, 'rocks': models_learn.rocks}
        eval_to_grid_model_names = {"avoid": models.surveillance, "refuel": models.refuel,
                                          'obstacle': models.obstacle, "intercept": models.intercept,
                                          'evade': models.evade, 'rocks': models.rocks}
        model = experiment_to_grid_model_names[args.grid_model]
        eval_model = eval_to_grid_model_names[args.grid_model]
        model_constants = list(inspect.signature(model).parameters.keys())
        if args.constants is None and len(model_constants) > 0:
            raise RuntimeError("Model constants {} defined, but not given by command line".format(",".join(model_constants)))
        constants = dict(item.split('=') for item in args.constants.split(","))
        learn_input = model(**constants)
        eval_input = eval_model(**constants)
        input = learn_input
    else:
        input = ManualInput(args.prism, args.prop, args.constants)
        eval_input = None
        constants = dict(item.split('=') for item in args.constants.split(","))


    if args.load_winning_region:
        logger.info("Load winning region...")
        winning_region, preamble = stormpy.pomdp.BeliefSupportWinningRegion.load_from_file(args.load_winning_region)
        for line in preamble.split('\n'):
            if line == "":
                continue
            if line.startswith("model hash: "):
                hash = int(line[12:])
        compute_shield = False
    else:
        winning_region = None
        compute_shield = not args.noshield

    initial = False
    logger.info("Loading problem definition....")
    learn_model,prism_program,raw_formula = build_model(input)
    if eval_input:
        eval_model,_,_ = build_model(eval_input)
    else:
        eval_model,_,_ = learn_model
    logger.info(model)

    #if model.hash() != hash:
    #    raise RuntimeError("Winning Region does not agree with Model")

    if compute_shield:
        winning_region = compute_winning_region(learn_model, raw_formula, initial)

    if winning_region is not None:
        otf_shield = construct_otf_shield(learn_model, winning_region)
    elif args.noshield:
        otf_shield = NoShield()
    else:
        logger.warning("No winning region: Shielding disabled.")
        otf_shield = NoShield()

    if args.load_winning_region:
        videoname = os.path.splitext(os.path.basename(args.load_winning_region))[0]
    else:
        constant_values = "-".join(constants.values())
        if compute_shield:
            videoname = f"{args.grid_model}-{constant_values}-computed-shield"
        else:
            videoname = f"{args.grid_model}-{constant_values}-noshield"

    tracker = Tracker(learn_model, otf_shield)
    if args.video_path:
        renderer = Plotter(prism_program, input.annotations, eval_model)
        if input.ego_icon is not None:
            renderer.load_ego_image(input.ego_icon.path, (0.6 / renderer._maxX))
        if args.title:
            renderer.set_title(args.title)
        recorder = VideoRecorder(renderer, only_keep_finishers=args.finishers_only)
        stats_recorder = StatsRecorder(only_keep_finishers=True)
        output_path = args.video_path
    elif args.stats_path:
        recorder = StatsRecorder(only_keep_finishers=args.finishers_only)
        output_path = args.stats_path
    else:
        logger.info("No video path set, rendering disabled.")
        output_path = None
        recorder = LoggingRecorder(only_keep_finishers=args.finishers_only)

    hyper_param = {}
    if hasattr(args,'network_size'):
        hyper_param.update({'network_size':args.__getattribute__('network_size')})
    if hasattr(args,'alpha'):
        hyper_param.update({'alpha':args.__getattribute__('alpha')})
    obs_type = args.obs_level
    valuations = args.valuations
    if hasattr(args,'fname'):
        result_fname = f""+args.fname
    else:
        result_fname = f"_Hyper_Param_Size_{args.network_size}_{obs_type}_valuations" if valuations else f"_{obs_type}"
    learn_executor = TF_Environment(learn_model,tracker,obs_length=1,maxsteps=args.maxsteps,obs_type=obs_type,valuations=valuations,goal_value=args.goal_value)
    eval_executor = TF_Environment(eval_model,tracker,obs_length=1,maxsteps=args.maxsteps,obs_type=obs_type,valuations=valuations,goal_value=args.goal_value)
    if hasattr(args,'switch_shield'):
        learn_executor.set_shield_switch(args.shield_episode,args.switch_shield)
        eval_executor.set_shield_switch(-1,args.switch_shield)
    if hasattr(args,'fixed_policy'):
        learn_executor.set_fixed_policy(args.fixed_policy_p)
        eval_executor.set_fixed_policy(args.fixed_policy_p)
    print("Starting RL:\n")
    print(f"{videoname}{result_fname}")
    G0,episodes = learn_executor.simulate_deep_RL(recorder,total_nr_runs=args.max_runs, eval_interval=args.eval_interval,eval_episodes=args.eval_episodes,eval_env= eval_executor,agent_arg=args.learning_method,hyper_param=hyper_param)
    np.savetxt(f"{output_path}/{videoname}{result_fname}.csv",np.array(G0),delimiter=' ')
    np.savetxt(f"{output_path}/{videoname}{result_fname}_Episodes.csv",np.array(episodes),delimiter=' ')
    recorder.save(output_path, f"{videoname}{result_fname}")

if __name__ == "__main__":
    main()
