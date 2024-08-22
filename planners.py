import backoff
import json
import openai

from collections import namedtuple
from pydantic_generator import PydanticModelGenerator
from utils import openai_client
from juliacall import Main as jl

# Initialize Julia and load PDDL package
jl.seval('using PDDL, SymbolicPlanners')

PlannerResult = namedtuple("PlannerResult", ["plan_pddl", "plan_json", "task_pddl"])

class BasePlanner:
    def run_planner(self, task_nl, domain_nl, domain_pddl) -> PlannerResult:
        raise NotImplementedError

    def set_context(self, context):
        self.context = context

    def _create_prompt(self, task_nl, domain_nl):
        pass

    def _query_llm(self, prompt_text, domain_pddl = None):

        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def completions_with_backoff(**kwargs):
            return openai_client.beta.chat.completions.parse(**kwargs)

        response_format = None
        if domain_pddl:
            model_generator = PydanticModelGenerator(domain_pddl)
            response_format = model_generator.create_response_model()

        server_cnt = 0
        result_text = ""
        while server_cnt < 10:
            try:

                completions_args = {
                    'model': "gpt-4o-2024-08-06",
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

    def run_planner(self, task_nl, domain_nl, domain_pddl) -> PlannerResult:
        
        prompt = self._create_prompt(task_nl, domain_nl)
        plan_json = self._query_llm(prompt, domain_pddl)
        plan_pddl = self._translate_json_to_pddl(plan_json)

        res = PlannerResult(
            plan_pddl=plan_pddl, 
            plan_json=plan_json,
            task_pddl=None
            )

        return res

    def _translate_json_to_pddl(self, json_structured_plan: str):
        plan_pddl = []

        plan_dict = json.loads(json_structured_plan)
        
        for step in plan_dict["steps"]:
            action_name = step["action_name"]
            arguments = [value for key, value in step.items() if key != "action_name"]
            plan_pddl.append(f"({action_name} {' '.join(arguments)})")
        
        return "\n".join(plan_pddl)

class BaseLlmPddlPlanner(BasePlanner):

    def run_planner(self, task_nl, domain_nl, domain_pddl) -> PlannerResult:
        
        prompt = self._create_prompt(task_nl, domain_nl)
        task_pddl = self._query_llm(prompt)
        plan_pddl = self._run_symbolic_planner(domain_pddl, task_pddl)

        res = PlannerResult(
            plan_pddl=plan_pddl, 
            plan_json=None,
            task_pddl=task_pddl
            )

        return res

    def _run_symbolic_planner(self, domain_pddl_text, problem_pddl_text):

        # plan
        domain = jl.PDDL.parse_domain(domain_pddl_text)
        problem = jl.PDDL.parse_problem(problem_pddl_text)
        planner = jl.SymbolicPlanners.AStarPlanner(jl.SymbolicPlanners.HAdd())
        if jl.isnothing(jl.PDDL.get_constraints(problem)):
            sol = planner(domain, problem)
        else:
            state = jl.PDDL.initstate(domain, problem)
            spec = jl.SymbolicPlanners.StateConstrainedGoal(problem)
            sol = planner(domain, state, spec)

        sol_str = "\n".join([jl.PDDL.write_pddl(a) for a in sol])
        return sol_str

class LlmIcPddlPlanner(BaseLlmPddlPlanner):
    """
    Our method:
        context: (task natural language, task problem PDDL)
        Condition on the context (task description -> task problem PDDL),
        LLM will be asked to provide the problem PDDL of a new task description.
        Then, we use a planner to find a correct solution, and translate
        that back to natural language.
    """

    def _create_prompt(self, task_nl, domain_nl):
        # our method (LM+P), create the problem PDDL given the context
        context_nl, context_pddl, context_sol = self.context
        prompt = f"I want you to solve planning problems. " + \
                 f"An example planning problem is: \n {context_nl} \n" + \
                 f"The problem PDDL file to this problem is: \n {context_pddl} \n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"Provide me with the problem PDDL file that describes " + \
                 f"the new planning problem directly without further explanations? " + \
                 f"Only return the text of the PDDL file. " + \
                 f"Do not include any code block delimiters and do not return anything else."
        return prompt

class LlmPddlPlanner(BaseLlmPddlPlanner):
    """
    Baseline method:
        Same as ours, except that no context is given. In other words, the LLM
        will be asked to directly give a problem PDDL file without any context.
    """

    def _create_prompt(self, task_nl, domain_nl):
        # Baseline 3 (LM+P w/o context), no context, create the problem PDDL
        prompt = f"{domain_nl} \n" + \
                 f"Now consider a planning problem. " + \
                 f"The problem description is: \n {task_nl} \n" + \
                 f"Provide me with the problem PDDL file that describes " + \
                 f"the planning problem directly without further explanations?" +\
                 f"Keep the domain name consistent in the problem PDDL. Only return the text of the PDDL file. Do not return anything else."
        return prompt

class LlmPlanner(BaseLlmPlanner):
    """
    Baseline method:
        The LLM will be asked to directly give a plan based on the task description.
    """

    def _create_prompt(self, task_nl, domain_nl):
        # Baseline 1 (LLM-as-P): directly ask the LLM for plan
        prompt = f"{domain_nl} \n" + \
                 f"Now consider a planning problem. " + \
                 f"The problem description is: \n {task_nl} \n" + \
                 f"Can you provide a correct plan, in the way of a " + \
                 f"sequence of behaviors, to solve the problem?"
        return prompt

class LlmIcPlanner(BaseLlmPlanner):
    """
    Baseline method:
        The LLM will be asked to directly give a plan based on the task description.
    """

    def _create_prompt(self, task_nl, domain_nl):
        # Baseline 2 (LLM-as-P with context): directly ask the LLM for plan
        context_nl, context_pddl, context_sol = self.context
        prompt = f"{domain_nl} \n" + \
                 f"An example planning problem is: \n {context_nl} \n" + \
                 f"A plan for the example problem is: \n {context_sol} \n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"Can you provide a correct plan, in the way of a " + \
                 f"sequence of behaviors, to solve the problem?"
        return prompt

class LlmSbSPlanner(LlmPlanner):
    """
    Baseline method:
        The LLM will be asked to directly give a plan based on the task description.
    """

    def _create_prompt(self, task_nl, domain_nl):
        # Baseline 1 (LLM-as-P): directly ask the LLM for plan

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