import stormpy as sp
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import stormpy.pomdp

def compute_winning_region(model, formula, initial=True):
    options = sp.pomdp.IterativeQualitativeSearchOptions()
    model = sp.pomdp.prepare_pomdp_for_qualitative_search_Double(model, formula)
    solver = sp.pomdp.create_iterative_qualitative_search_solver_Double(model, formula, options)
    if initial:
        solver.compute_winning_policy_for_initial_states(100)
    else:
        solver.compute_winning_region(100)
    return solver.last_winning_region

def construct_otf_shield(model, winning_region):
    return sp.pomdp.BeliefSupportWinningRegionQueryInterfaceDouble(model, winning_region)

def build_pomdp(program, formula):
    options = stormpy.BuilderOptions([formula])
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()
    options.set_build_all_reward_models()
    options.set_build_observation_valuations()
    # logger.debug("Start building the POMDP")
    return sp.build_sparse_model_with_options(program, options)

class ManualInput:
    def __init__(self, path, prop, constants):
        self.path = path
        self.properties = [prop]
        self.constants = constants


def build_shield(path,prop,constants):
    input = ManualInput(path,prop,constants)
    # constants = dict(item.split('=') for item in constants.split(","))
    prism_program = sp.parse_prism_program(input.path)
    prop = sp.parse_properties_for_prism_program(input.properties[0], prism_program)[0]
    prism_program, props = stormpy.preprocess_symbolic_input(prism_program, [prop], input.constants)
    prop = props[0]
    prism_program = prism_program.as_prism_program()
    raw_formula = prop.raw_formula

    model = build_pomdp(prism_program, raw_formula)
    model = sp.pomdp.make_canonic(model)
    winning_region = compute_winning_region(model, raw_formula, False)
    return construct_otf_shield(model, winning_region)




