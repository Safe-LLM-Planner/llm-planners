import os
import argparse
from collections import namedtuple
import time

from .planners import available_planners
from .pydantic_generator import available_pydantic_generators
from .config import DEFAULT_PYD_GENERATORS

PlannerPydModelTuple = namedtuple("PlannerPydModelTuple", ["planner", "pyd_gen"])

def method_tuple(s):
    parts = s.split(',')

    # Handle the case where only one value is provided
    if len(parts) == 1:
        planner_name = parts[0].strip()
        method = PlannerPydModelTuple(planner_name, DEFAULT_PYD_GENERATORS[planner_name])
    elif len(parts) == 2:
        method = PlannerPydModelTuple(parts[0].strip(), parts[1].strip())
    else:
        raise argparse.ArgumentTypeError("Tuple must have one or two elements")
    
    # Validate each element
    validate_planner(method.planner)
    validate_pydantic_generator(method.pyd_gen)
    
    return method

def validate_planner(name: str):
    if name not in available_planners.keys():
        raise argparse.ArgumentTypeError(f"Invalid value '{name}' for planner. Must be one of {available_planners.keys()}")

def validate_pydantic_generator(name: str):
    if name not in available_pydantic_generators.keys():
        raise argparse.ArgumentTypeError(f"Invalid value '{name}' for pydantic response model generator. Must be one of {available_pydantic_generators.keys()}")

method_tuple_help_text = (
            f"Provide one or more tuples in the format 'planner,pyd_gen'. Valid planners: {list(available_planners.keys())}. "
            f"Valid Pydantic generators: {list(available_pydantic_generators.keys())}. If only one value is provided, the default for the second value depends on the planner: '{DEFAULT_PYD_GENERATORS}'."
    )

def create_parser():
    parser = argparse.ArgumentParser(description="LLM Planners")
    parser.add_argument('method', type=method_tuple, help=method_tuple_help_text)
    parser.add_argument('domain', type=str, help="Domain path prefix.")
    parser.add_argument('example', type=str, help="Example path prefix.")
    parser.add_argument('problem', type=str, help="Problem path prefix.")
    parser.add_argument('output', type=str, help="Output directory.")

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Domain
    filename = f"{args.domain}.nl"
    with open(filename, 'r') as f:
        domain_nl = f.read()  

    filename = f"{args.domain}.pddl"
    with open(filename, 'r') as f:
        domain_pddl = f.read()  

    domain_name = os.path.basename(args.domain)

    # Example
    context = {}
    if args.method.planner == "llm_ic_pddl":
        keys = ["init_nl", "init_pddl", "goal_nl", "goal_pddl", "constraints_nl", "constraints_pddl"]
    elif args.method.planner == "llm_ic":
        keys = ["init_nl", "goal_nl", "constraints_nl", "sol"]

    for k in keys:
        ext = k.replace("_",".")
        filename = f"{args.example}.{ext}"
        with open(filename, 'r') as f:
            context[k] = f.read()    

    # Problem
    filename = f"{args.problem}.init.nl"
    with open(filename, 'r') as f:
        init_nl = f.read()  
        
    filename = f"{args.problem}.goal.nl"
    with open(filename, 'r') as f:
        goal_nl = f.read()  
        
    filename = f"{args.problem}.constraints.nl"
    with open(filename, 'r') as f:
        constraints_nl = f.read()

    problem_name = os.path.basename(args.problem)

    planner = available_planners[args.method.planner]
    planner.set_context(context, domain_name, problem_name)
    planner.set_response_model_generator(args.method.pyd_gen)

    start_time = time.time()
    planner_result = planner.run_planner(init_nl, goal_nl, constraints_nl, domain_nl, domain_pddl)
    end_time = time.time()

    if (planner_result.plan_json is not None):
        plan_json_file_name = os.path.join(args.output,f"{problem_name}.json")
        with open(plan_json_file_name, "w") as f:
            f.write(planner_result.plan_json)

    if (planner_result.task_pddl is not None):
        produced_task_pddl_file_name = os.path.join(args.output,f"{problem_name}.pddl")
        with open(produced_task_pddl_file_name, "w") as f:
            f.write(planner_result.task_pddl)

    if (planner_result.plan_pddl is not None):
        plan_pddl_file_name = os.path.join(args.output,f"{problem_name}.plan.pddl")
        with open(plan_pddl_file_name, "w") as f:
            f.write(planner_result.plan_pddl)

    print(f"Solving time {end_time - start_time} s.")
    return planner_result