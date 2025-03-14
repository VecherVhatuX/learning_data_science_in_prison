import os
import time
import random
from subprocess import run, CalledProcessError
import click
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_items(items):
    return random.sample(items, len(items))

def split_by_label(items):
    return (
        [item for item in items if item['label'] == 1],
        [item for item in items if item['label'] == 0]
    )

def process_epoch(items, epoch):
    return (split_by_label(randomize_items(items)), epoch + 1)

def display_env(info):
    return f"Environment details: {dict(os.environ)}\n{info}"

def install_dependency(dependency):
    try:
        result = run(["pip", "install", dependency], text=True, capture_output=True, check=True)
        return result.stdout
    except CalledProcessError as e:
        return e.stderr

def execute_shell(cmd):
    if not cmd:
        return "", "Command is empty or invalid."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools():
    return [
        Tool(name="EnvViewer", func=display_env, description="Displays the current environment settings."),
        Tool(name="PackageManager", func=install_dependency, description="Installs required packages.")
    ]

def log_execution(cmd):
    logger.info(f"Running command: {cmd}")
    output, error = execute_shell(cmd)
    if error:
        logger.error(f"Command failed: {error}")
    else:
        logger.success(f"Command succeeded: {output}")
    return not error

def retry_execution(agent, cmd, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Maximum retries reached. Aborting.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if log_execution(cmd):
        logger.success("Command executed successfully!")
        return
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing missing packages.")
    time.sleep(5)
    retry_execution(agent, cmd, attempt + 1, max_attempts)

def execute_with_retries(cmd, max_attempts):
    tools = initialize_tools()
    agent = create_agent(tools=tools)
    retry_execution(agent, cmd, 0, max_attempts)

def time_execution(func):
    @wraps(func)
    def timed_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return timed_func

@time_execution
def start_process(cmd, max_attempts):
    logger.info("Initiating process...")
    execute_with_retries(cmd, max_attempts)

def log_command(cmd):
    try:
        with open("command_history.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except IOError as e:
        logger.error(f"Failed to log command: {e}")

def countdown(seconds):
    if seconds > 0:
        logger.info(f"Timer: {seconds} seconds remaining")
        time.sleep(1)
        countdown(seconds - 1)
    else:
        logger.success("Timer finished!")

def run_command(cmd, max_attempts=5, timer_duration=0):
    log_command(cmd)
    if timer_duration > 0:
        countdown(timer_duration)
    start_process(cmd, max_attempts)

if __name__ == "__main__":
    click.command()(run_command)()