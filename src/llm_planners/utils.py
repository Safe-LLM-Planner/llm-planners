import openai
import os
import subprocess

def postprocess(x):
    return x.strip()

def load_openai_key():
    openai_keys_file = os.path.join(os.getcwd(), "keys/openai_keys.txt")
    
    try:
        with open(openai_keys_file, "r") as f:
            context = f.read()
        openai_api_key = context.strip().split('\n')[0]
        return openai_api_key
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: The file 'keys/openai_keys.txt' was not found in the current working directory: {os.getcwd()}. "
            "Please ensure the file exists and contains the OpenAI API key."
        )

openai_client = openai.OpenAI(api_key=load_openai_key())