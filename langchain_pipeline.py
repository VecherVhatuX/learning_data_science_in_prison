import os
import time
import random
from subprocess import run, CalledProcessError
import click
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_data(data):
    return random.sample(data, len(data))

def split_data(data):
    return (
        [item for item in data if item['label'] == 1],
        [item for item in data if item['label'] == 0]
    )

def process_epoch(data, epoch):
    return (split_data(randomize_data(data)), epoch + 1)

def display_environment(info):
    return f"Environment details: {dict(os.environ)}\n{info}"

def install_package(package):
    try:
        result = run(["pip", "install", package], text=True, capture_output=True, check=True)
        return result.stdout
    except CalledProcessError as e:
        return e.stderr

def execute_command(command):
    if not command:
        return "", "Command is empty or invalid."
    result = run(command, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools():
    return [
        Tool(name="EnvDisplay", func=display_environment, description="Shows the current environment settings."),
        Tool(name="PackageInstaller", func=install_package, description="Installs necessary packages.")
    ]

def log_command_execution(command):
    logger.info(f"Executing command: {command}")
    output, error = execute_command(command)
    if error:
        logger.error(f"Execution failed: {error}")
    else:
        logger.success(f"Execution successful: {output}")
    return not error

def retry_execution(agent, command, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Max retries reached. Stopping.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if log_command_execution(command):
        logger.success("Command executed successfully!")
        return
    agent.run("Check environment variables and dependencies.")
    agent.run("Attempt to fix environment issues by installing missing packages.")
    time.sleep(5)
    retry_execution(agent, command, attempt + 1, max_attempts)

def execute_with_agent(command, max_attempts):
    tools = initialize_tools()
    agent = create_agent(tools=tools)
    retry_execution(agent, command, 0, max_attempts)

def time_execution(func):
    @wraps(func)
    def timed_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return timed_wrapper

@time_execution
def start_process(command, max_attempts):
    logger.info("Starting process...")
    execute_with_agent(command, max_attempts)

def log_command(command):
    try:
        with open("command_log.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
    except IOError as e:
        logger.error(f"Logging failed: {e}")

def start_countdown(seconds):
    if seconds > 0:
        logger.info(f"Countdown: {seconds} seconds left")
        time.sleep(1)
        start_countdown(seconds - 1)
    else:
        logger.success("Countdown complete!")

def run_command(command, max_attempts=5, countdown_time=0):
    log_command(command)
    if countdown_time > 0:
        start_countdown(countdown_time)
    start_process(command, max_attempts)

if __name__ == "__main__":
    click.command()(run_command)()