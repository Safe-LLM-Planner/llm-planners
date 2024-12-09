import backoff
import json
import openai
import subprocess
import os

from collections import namedtuple
from pydantic_generator import available_pydantic_generators
from utils import openai_client
from config import OPENAI_MODEL
from juliacall import Main as jl
from pddl_constraints_translator import PDDLConstraintsTranslator

# Initialize Julia and load PDDL package
jl.seval('using PDDL, SymbolicPlanners')

PlannerResult = namedtuple("PlannerResult", ["plan_pddl", "plan_json", "task_pddl"])

def run_symbolic_planner_jl(domain_pddl_text, problem_pddl_text):
    # plan
    domain = jl.PDDL.parse_domain(domain_pddl_text)
    problem = jl.PDDL.parse_problem(problem_pddl_text)
    planner = jl.SymbolicPlanners.ForwardPlanner()
    # planner = jl.SymbolicPlanners.AStarPlanner(jl.SymbolicPlanners.HAdd())
    if jl.isnothing(jl.PDDL.get_constraints(problem)):
        sol = planner(domain, problem)
    else:
        state = jl.PDDL.initstate(domain, problem)
        spec = jl.SymbolicPlanners.StateConstrainedGoal(problem)
        sol = planner(domain, state, spec)

    sol_str = "\n".join([jl.PDDL.write_pddl(a) for a in sol])
    return sol_str

def run_fast_downward_planner(domain_pddl_text, problem_pddl_text, optimality=False):

    # non-optimal search strategy
    FAST_DOWNWARD_NONOPTIMAL_SEARCH = "eager_greedy([add()])"

    # optimal search strategy
    FAST_DOWNWARD_OPTIMAL_SEARCH = "astar(blind())"

    translator = PDDLConstraintsTranslator()
    domain_pddl_text = translator.translate_domain(domain_pddl_text, problem_pddl_text)

    time_limit = 200
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    domain_pddl_file = os.path.join(tmp_dir, "domain.pddl")
    problem_pddl_file_name = os.path.join(tmp_dir, "problem.pddl")
    sas_file_name = os.path.join(tmp_dir, "problem.sas")
    plan_file_name = os.path.join(tmp_dir, "plan.pddl")
    
    
    try:
        # Write domain and problem PDDL content to temporary files
        with open(domain_pddl_file, "w") as domain_file:
            domain_file.write(domain_pddl_text)
        with open(problem_pddl_file_name, "w") as problem_file:
            problem_file.write(problem_pddl_text)
        
        # Construct the Fast Downward command
        run_command = [
            "python", "./downward/fast-downward.py",
        ]

        run_command += [
            "--search-time-limit", str(time_limit),
            "--plan-file", plan_file_name,
            "--sas-file", sas_file_name,
            domain_pddl_file,
            problem_pddl_file_name
        ]

        if optimality:
            run_command += ["--search", FAST_DOWNWARD_OPTIMAL_SEARCH]
        else:
            run_command += ["--search", FAST_DOWNWARD_NONOPTIMAL_SEARCH]

        # print(" ".join(run_command))

        # Run the command
        result = subprocess.run(run_command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Planner failed: {result.stderr}")
        
        # print("Planner output:")
        # print(result.stdout)
        
        # Check if the plan file was created
        if os.path.exists(plan_file_name):
            with open(plan_file_name, "r") as plan_file:
                plan = plan_file.read()
            # print("Plan found:")
            # print(plan)
            return translator.translate_plan_back(plan)
        else:
            # print("No plan found.")
            return ""
    
    except Exception as e:
        print(f"Error during planning: {e}")
    
    # finally:
    #     # Cleanup temporary files
    #     if os.path.exists(domain_pddl_file):
    #         os.remove(domain_pddl_file)
    #     if os.path.exists(problem_pddl_file_name):
    #         os.remove(problem_pddl_file_name)
    #     if os.path.exists(sas_file_name):
    #         os.remove(sas_file_name)
    #     if os.path.exists(plan_file_name):
    #         os.remove(plan_file_name)

class BasePlanner:
    def run_planner(self, init_nl, goal_nl, constraints_nl, domain_nl, domain_pddl) -> PlannerResult:
        raise NotImplementedError

    def set_context(self, context, domain_name, task_name):
        self.context = context
        self.domain_name = domain_name
        self.task_name = task_name

    def set_response_model_generator(self, model_generator_name: str):
        if model_generator_name:
            self.model_generator = available_pydantic_generators[model_generator_name]
        else:
            self.model_generator = None

    def _load_prompt_templates(self):
        raise NotImplementedError

    def _query_llm(self, prompt_text, domain_pddl = None):

        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def completions_with_backoff(**kwargs):
            return openai_client.beta.chat.completions.parse(**kwargs)

        response_format = None
        if self.model_generator:
            if domain_pddl:
                response_format = self.model_generator(domain_pddl).create_response_model()
            else:
                raise ValueError("Cannot generate Pydantic model if the pddl domain is not given")

        server_cnt = 0
        result_text = ""
        while server_cnt < 10:
            try:
                completions_args = {
                    'model': OPENAI_MODEL,
                    'temperature': 0.0,
                    'top_p': 1,
                    'frequency_penalty': 0,
                    'presence_penalty': 0,
                    'messages': [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_text},
                    ]
                }
                if response_format:
                    completions_args['response_format'] = response_format
                response = completions_with_backoff(**completions_args)
                result_text = response.choices[0].message.content
                break
            except Exception as e:
                server_cnt += 1
                print(e)
        return result_text

class BaseLlmPlanner(BasePlanner):

    def run_planner(self, init_nl, goal_nl, constraints_nl, domain_nl, domain_pddl) -> PlannerResult:
        
        prompt = self._create_prompt(init_nl, goal_nl, constraints_nl, domain_nl)
        plan_json = self._query_llm(prompt, domain_pddl)

        res = PlannerResult(
            plan_pddl=None, 
            plan_json=plan_json,
            task_pddl=None
            )

        return res

    def _load_prompt_templates(self):
        if hasattr(self, 'name'):
            with open(f'prompt_templates/{self.name}.prompt', 'r') as file:
                self.prompt_template = file.read()
        else:
            raise ValueError("Planner name not defined")

    # def _translate_json_to_pddl(self, json_structured_plan: str):
    #     plan_pddl = []

    #     plan_dict = json.loads(json_structured_plan)
        
    #     for step in plan_dict["steps"]:
    #         action_name = step["action_name"]
    #         arguments = [value for key, value in step.items() if key != "action_name"]
    #         plan_pddl.append(f"({action_name} {' '.join(arguments)})")
        
    #     return "\n".join(plan_pddl)

class BaseLlmPddlPlanner(BasePlanner):

    def run_planner(self, init_nl, goal_nl, constraints_nl, domain_nl, domain_pddl) -> PlannerResult:
        
        init_prompt = self._create_init_prompt(init_nl, domain_nl, domain_pddl)
        init_pddl = self._query_llm(init_prompt)
        init_pddl = init_pddl.strip("`")

        goal_prompt = self._create_goal_prompt(goal_nl, init_pddl, domain_nl, domain_pddl)
        goal_pddl = self._query_llm(goal_prompt)
        goal_pddl = goal_pddl.strip("`")

        constraints_prompt = self._create_constraints_prompt(constraints_nl, init_pddl, domain_nl, domain_pddl)
        constraints_pddl = self._query_llm(constraints_prompt)
        constraints_pddl = constraints_pddl.strip("`")

        task_pddl = self._compose_task_pddl(init_pddl, goal_pddl, constraints_pddl)

        try:
            plan_pddl = self._run_symbolic_planner(domain_pddl, task_pddl)
        except:
            plan_pddl = "; symbolic planner error"

        res = PlannerResult(
            plan_pddl=plan_pddl, 
            plan_json=None,
            task_pddl=task_pddl
            )

        return res

    def _compose_task_pddl(self, init_pddl, goal_pddl, constraints_pddl) -> str:
        problem_name_pddl = f"(problem {self.domain_name}-{self.task_name})"
        domain_name_pddl = f"(:domain {self.domain_name})"
        return f"(define {problem_name_pddl} {domain_name_pddl} {init_pddl} {goal_pddl} {constraints_pddl})"

    def _run_symbolic_planner(self, domain_pddl_text, problem_pddl_text):
        return run_symbolic_planner_jl(domain_pddl_text, problem_pddl_text)

    def _load_prompt_templates(self):
        if hasattr(self, 'name'):
            with open(f'prompt_templates/{self.name}_init.prompt', 'r') as file:
                self.init_prompt_template = file.read()
            with open(f'prompt_templates/{self.name}_goal.prompt', 'r') as file:
                self.goal_prompt_template = file.read()
            with open(f'prompt_templates/{self.name}_constraints.prompt', 'r') as file:
                self.constraints_prompt_template = file.read()
        else:
            raise ValueError("Planner name not defined")

class LlmIcPddlPlanner(BaseLlmPddlPlanner):

    def __init__(self):
        self.name = "llm_ic_pddl"
        self._load_prompt_templates()

    def _create_init_prompt(self, init_nl, domain_nl, domain_pddl) -> str:
        prompt = self.init_prompt_template.format(
            domain_nl=domain_nl,
            context_init_nl = self.context["init_nl"],
            context_init_pddl = self.context["init_pddl"],
            init_nl=init_nl
        )
        return prompt

    def _create_goal_prompt(self, goal_nl, init_pddl, domain_nl, domain_pddl) -> str:
        prompt = self.goal_prompt_template.format(
            domain_nl=domain_nl,
            context_goal_nl = self.context["goal_nl"],
            context_goal_pddl = self.context["goal_pddl"],
            init_pddl=init_pddl,
            goal_nl=goal_nl
        )
        return prompt

    def _create_constraints_prompt(self, constraints_nl, init_pddl, domain_nl, domain_pddl) -> str:
        prompt = self.constraints_prompt_template.format(
            domain_nl=domain_nl,
            context_constraints_nl = self.context["constraints_nl"],
            context_constraints_pddl = self.context["constraints_pddl"],
            init_pddl=init_pddl,
            constraints_nl=constraints_nl
        )
        return prompt

class LlmPddlPlanner(BaseLlmPddlPlanner):

    def __init__(self):
        self.name = "llm_pddl"
        self._load_prompt_templates()

    def _create_init_prompt(self, init_nl, domain_nl, domain_pddl) -> str:
        prompt = self.init_prompt_template.format(
            domain_nl=domain_nl,
            domain_pddl=domain_pddl,
            init_nl=init_nl
        )
        return prompt

    def _create_goal_prompt(self, goal_nl, init_pddl, domain_nl, domain_pddl) -> str:
        prompt = self.goal_prompt_template.format(
            domain_nl=domain_nl,
            domain_pddl=domain_pddl,
            init_pddl=init_pddl,
            goal_nl=goal_nl
        )
        return prompt

    def _create_constraints_prompt(self, constraints_nl, init_pddl, domain_nl, domain_pddl) -> str:
        prompt = self.constraints_prompt_template.format(
            domain_nl=domain_nl,
            domain_pddl=domain_pddl,
            init_pddl=init_pddl,
            constraints_nl=constraints_nl
        )
        return prompt

class LlmPlanner(BaseLlmPlanner):

    def __init__(self):
        self.name = "llm"
        self._load_prompt_templates()

    def _create_prompt(self, init_nl, goal_nl, constraints_nl, domain_nl):
        prompt = self.prompt_template.format(
            domain_nl=domain_nl,
            init_nl=init_nl,
            goal_nl=goal_nl,
            constraints_nl=constraints_nl
        )
        return prompt

class LlmIcPlanner(BaseLlmPlanner):

    def __init__(self):
        self.name = "llm_ic"
        self._load_prompt_templates()

    def _create_prompt(self, init_nl, goal_nl, constraints_nl, domain_nl):
        prompt = self.prompt_template.format(
            domain_nl=domain_nl,
            context_init_nl=self.context["init_nl"],
            context_goal_nl=self.context["goal_nl"],
            context_constraints_nl=self.context["constraints_nl"],
            context_sol=self.context["sol"],
            init_nl=init_nl,
            goal_nl=goal_nl,
            constraints_nl=constraints_nl
        )
        return prompt

class LlmSbSPlanner(LlmPlanner):

    def _create_prompt(self, task_nl, domain_nl):

        prompt = super()._create_prompt(task_nl, domain_nl)
        prompt += " \nPlease think step by step."
        return prompt

# class LlmTotPlanner(BasePlanner):
#     """
#     Tree of Thoughts planner
#     """

#     def run_planner(self, task_nl, domain_nl, domain_pddl, time_left=200, max_depth=2) -> PlannerResult:
#         context = self.context
#         from queue import PriorityQueue
#         start_time = time.time()
#         plan_queue = PriorityQueue()
#         plan_queue.put((0, ""))
#         while time.time() - start_time < time_left and not plan_queue.empty():
#             priority, plan = plan_queue.get()
#             # print (priority, plan)
#             steps = plan.split('\n')
#             if len(steps) > max_depth:
#                 return "", None
#             candidates_prompt = self._create_llm_tot_ic_prompt(task_nl, domain_nl, context, plan)
#             candidates = self._query_llm(candidates_prompt).strip()
#             print (candidates)
#             lines = candidates.split('\n')
#             for line in lines:
#                 if time.time() - start_time > time_left:
#                     break
#                 if len(line) > 0 and '->' in line:
#                     new_plan = plan + "\n" + line
#                     value_prompt = self._create_llm_tot_ic_value_prompt(task_nl, domain_nl, context, new_plan)
#                     answer = self._query_llm(value_prompt).strip().lower()
#                     print(new_plan)
#                     print("Response \n" + answer)

#                     if "reached" in answer:
#                         return new_plan, None

#                     if "impossible" in answer:
#                         continue

#                     if "answer: " in answer:
#                         answer = answer.split("answer: ")[1]

#                     try:
#                         score = float(answer)
#                     except ValueError:
#                         continue

#                     if score > 0:
#                         new_priority = priority + 1 / score
#                         plan_queue.put((new_priority, new_plan))

#         return ""

#     def _create_llm_tot_ic_prompt(self, task_nl, domain_nl, context, plan):
#         context_nl, context_pddl, context_sol = context
#         prompt = f"Given the current state, provide the set of feasible actions and their corresponding next states, using the format 'action -> state'. \n" + \
#                  f"Keep the list short. Think carefully about the requirements of the actions you select and make sure they are met in the current state. \n" + \
#                  f"Start with actions that are most likely to make progress towards the goal. \n" + \
#                  f"Only output one action per line. Do not return anything else. " + \
#                  f"Here are the rules. \n {domain_nl} \n\n" + \
#                  f"An example planning problem is: \n {context_nl} \n" + \
#                  f"A plan for the example problem is: \n {context_sol} \n" + \
#                  f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
#                  f"You have taken the following actions: \n {plan} \n"
#         # print(prompt)
#         return prompt

#     def _create_llm_tot_ic_value_prompt(self, task_nl, domain_nl, context, plan):
#         context_nl, context_pddl, context_sol = context
#         context_sure_1 = context_sol.split('\n')[0]
#         context_sure_2 = context_sol.split('\n')[0] + context_sol.split('\n')[1]
#         context_impossible_1 = '\n'.join(context_sol.split('\n')[1:])
#         context_impossible_2 = context_sol.split('\n')[-1]
#         '''
#         prompt = f"Evaluate if a given plan reaches the goal or is an optimal partial plan towards the goal (reached/sure/maybe/impossible). \n" + \
#                  f"Only answer 'reached' if the goal conditions are reached by the exact plan in the prompt. \n" + \
#                  f"Only answer 'sure' if you are sure that preconditions are satisfied for all actions in the plan, and the plan makes fast progress towards the goal. \n" + \
#                  f"Answer 'impossible' if one of the actions has unmet preconditions. \n" + \
#                  f"Here are the rules. \n {domain_nl} \n\n" + \
#                  f"Here are some example evaluations for the planning problem: \n {context_nl} \n\n " + \
#                  f"Plan: {context_sure_1} \n" + \
#                  f"Answer: Sure. \n\n" + \
#                  f"Plan: {context_sure_2} \n" + \
#                  f"Answer: Sure. \n\n" + \
#                  f"Plan: {context_sol} \n" + \
#                  f"Answer: Reached. \n\n" + \
#                  f"Plan: {context_impossible_1} \n" + \
#                  f"Answer: Impossible. \n\n" + \
#                  f"Plan: {context_impossible_2} \n" + \
#                  f"Answer: Impossible. \n\n" + \
#                  f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
#                  f"Evaluate the following partial plan as reached/sure/maybe/impossible. DO NOT RETURN ANYTHING ELSE. DO NOT TRY TO COMPLETE THE PLAN. \n" + \
#                  f"Plan: {plan} \n"
#         '''
#         prompt = f"Determine if a given plan reaches the goal or give your confidence score that it is an optimal partial plan towards the goal (reached/impossible/0-1). \n" + \
#                  f"Only answer 'reached' if the goal conditions are reached by the exact plan in the prompt. \n" + \
#                  f"Answer 'impossible' if one of the actions has unmet preconditions. \n" + \
#                  f"Otherwise,give a number between 0 and 1 as your evaluation of the partial plan's progress towards the goal. \n" + \
#                  f"Here are the rules. \n {domain_nl} \n\n" + \
#                  f"Here are some example evaluations for the planning problem: \n {context_nl} \n\n " + \
#                  f"Plan: {context_sure_1} \n" + \
#                  f"Answer: 0.8. \n\n" + \
#                  f"Plan: {context_sure_2} \n" + \
#                  f"Answer: 0.9. \n\n" + \
#                  f"Plan: {context_sol} \n" + \
#                  f"Answer: Reached. \n\n" + \
#                  f"Plan: {context_impossible_1} \n" + \
#                  f"Answer: Impossible. \n\n" + \
#                  f"Plan: {context_impossible_2} \n" + \
#                  f"Answer: Impossible. \n\n" + \
#                  f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
#                  f"Evaluate the following partial plan as reached/impossible/0-1. DO NOT RETURN ANYTHING ELSE. DO NOT TRY TO COMPLETE THE PLAN. \n" + \
#                  f"Plan: {plan} \n"

#         return prompt

available_planners = {
    "llm_ic_pddl"   : LlmIcPddlPlanner(),
    "llm_pddl"      : LlmPddlPlanner(),
    "llm"           : LlmPlanner(),
    "llm_ic"        : LlmIcPlanner(),
    "llm_stepbystep": LlmSbSPlanner(),
    # "llm_tot_ic"    : LlmTotPlanner() 
}