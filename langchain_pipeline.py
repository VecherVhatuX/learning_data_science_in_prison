import os
import time
import random
from subprocess import run, CalledProcessError
import typer
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def shuffle_and_categorize(items):
    shuffled = random.sample(items, len(items))
    return (
        [item for item in shuffled if item['label'] == 1],
        [item for item in shuffled if item['label'] == 0]
    )

def process_data(items, iteration):
    return (shuffle_and_categorize(items), iteration + 1)

def show_env_details(info):
    return f"Environment details: {dict(os.environ)}\n{info}"

def install_dependency(dependency):
    result = run(["pip", "install", dependency], text=True, capture_output=True, check=True)
    return result.stdout if not result.returncode else result.stderr

def run_command(command):
    if not command:
        return "", "Command is empty or invalid."
    result = run(command, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def setup_tools():
    return [
        Tool(name="EnvViewer", func=show_env_details, description="Displays the current environment settings."),
        Tool(name="PackageManager", func=install_dependency, description="Installs required packages.")
    ]

def log_command_execution(command):
    logger.info(f"Running command: {command}")
    output, error = run_command(command)
    if error:
        logger.error(f"Command failed: {error}")
        return False
    logger.success(f"Command succeeded: {output}")
    return True

def retry_command_execution(agent, command, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Maximum retries reached. Aborting.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if log_command_execution(command):
        logger.success("Command executed successfully!")
        return
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing missing packages.")
    time.sleep(5)
    retry_command_execution(agent, command, attempt + 1, max_attempts)

def execute_with_retries(command, max_attempts):
    tools = setup_tools()
    agent = create_agent(tools=tools)
    retry_command_execution(agent, command, 0, max_attempts)

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@measure_execution_time
def start_process(command, max_attempts):
    logger.info("Starting process...")
    execute_with_retries(command, max_attempts)

def log_command_to_file(command):
    try:
        with open("command_history.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
    except IOError as e:
        logger.error(f"Failed to log command: {e}")

def countdown_timer(seconds):
    if seconds > 0:
        logger.info(f"Timer: {seconds} seconds remaining")
        time.sleep(1)
        countdown_timer(seconds - 1)
    else:
        logger.success("Timer finished!")

def execute_command(command: str, max_attempts: int = 5, timer_duration: int = 0):
    log_command_to_file(command)
    if timer_duration > 0:
        countdown_timer(timer_duration)
    start_process(command, max_attempts)

if __name__ == "__main__":
    typer.run(execute_command)