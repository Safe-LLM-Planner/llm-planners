import backoff
import config
import openai
import os
import subprocess

def postprocess(x):
    return x.strip()

def get_cost(x):
    splitted = x.split()
    counter = 0
    found = False
    cost = 1e5
    for i, xx in enumerate(splitted):
        if xx == "cost":
            counter = i
            found = True
            break
    if found:
        cost = float(splitted[counter+2])
    return cost

def load_openai_key():
    openai_keys_file = os.path.join(os.getcwd(), "keys/openai_keys.txt")
    with open(openai_keys_file, "r") as f:
        context = f.read()
    openai_api_key = context.strip().split('\n')[0]
    return openai_api_key

openai_client = openai.OpenAI(api_key=load_openai_key())

def query_llm(prompt_text):
    server_cnt = 0
    result_text = ""
    while server_cnt < 10:
        try:
            @backoff.on_exception(backoff.expo, openai.RateLimitError)
            def completions_with_backoff(**kwargs):
                return openai_client.chat.completions.create(**kwargs)

            response = completions_with_backoff(
                model="gpt-4",
                temperature=0.0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text},
                ],
            )
            result_text = response.choices[0].message.content
            break
        except Exception as e:
            server_cnt += 1
            print(e)
    return result_text

def run_symbolic_planner(domain_pddl, task_pddl):

    os.makedirs("./tmp/",exist_ok=True)

    task_pddl_file_name = f"./tmp/task.pddl"
    with open(task_pddl_file_name, "w") as f:
        f.write(task_pddl)

    domain_pddl_file_name = f"./tmp/domain.pddl"
    with open(domain_pddl_file_name, "w") as f:
        f.write(domain_pddl)

    plan_file_name = f"./tmp/plan.pddl"

    command = ["julia", config.JULIA_PLANNER_SCRIPT, domain_pddl_file_name, task_pddl_file_name, plan_file_name]
    
    plan = ""

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Symbolic planner failed.")
    else:
        with open(plan_file_name, "r") as f:
            plan = f.read()
        
        os.remove(task_pddl_file_name)
        os.remove(domain_pddl_file_name)
        os.remove(plan_file_name)

    return plan