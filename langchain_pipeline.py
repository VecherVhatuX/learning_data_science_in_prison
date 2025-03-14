import os
import time
import random
from subprocess import run, CalledProcessError
import click
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def shuffle_data(data):
    return random.sample(data, len(data))

def categorize_data(data):
    return (
        [item for item in data if item['label'] == 1],
        [item for item in data if item['label'] == 0]
    )

def handle_epoch(data, epoch):
    return (categorize_data(shuffle_data(data)), epoch + 1)

def show_environment(info):
    return f"Environment details: {dict(os.environ)}\n{info}"

def add_package(package):
    try:
        result = run(["pip", "install", package], text=True, capture_output=True, check=True)
        return result.stdout
    except CalledProcessError as e:
        return e.stderr

def run_shell_command(command):
    if not command:
        return "", "Command is empty or invalid."
    result = run(command, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def setup_tools():
    return [
        Tool(name="EnvViewer", func=show_environment, description="Displays the current environment settings."),
        Tool(name="PackageManager", func=add_package, description="Installs required packages.")
    ]

def log_command_run(command):
    logger.info(f"Running command: {command}")
    output, error = run_shell_command(command)
    if error:
        logger.error(f"Command failed: {error}")
    else:
        logger.success(f"Command succeeded: {output}")
    return not error

def retry_command(agent, command, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Maximum retries reached. Aborting.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if log_command_run(command):
        logger.success("Command executed successfully!")
        return
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing missing packages.")
    time.sleep(5)
    retry_command(agent, command, attempt + 1, max_attempts)

def execute_with_retry(command, max_attempts):
    tools = setup_tools()
    agent = create_agent(tools=tools)
    retry_command(agent, command, 0, max_attempts)

def measure_time(func):
    @wraps(func)
    def timed_function(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return timed_function

@measure_time
def begin_process(command, max_attempts):
    logger.info("Initiating process...")
    execute_with_retry(command, max_attempts)

def record_command(command):
    try:
        with open("command_history.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
    except IOError as e:
        logger.error(f"Failed to log command: {e}")

def start_timer(seconds):
    if seconds > 0:
        logger.info(f"Timer: {seconds} seconds remaining")
        time.sleep(1)
        start_timer(seconds - 1)
    else:
        logger.success("Timer finished!")

def execute_command(command, max_attempts=5, timer_duration=0):
    record_command(command)
    if timer_duration > 0:
        start_timer(timer_duration)
    begin_process(command, max_attempts)

if __name__ == "__main__":
    click.command()(execute_command)()