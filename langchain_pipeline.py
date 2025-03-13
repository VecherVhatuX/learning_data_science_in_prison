import os
import time
import random
from subprocess import run, CalledProcessError
import click
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import reduce

# Randomly reorders the elements in the provided dataset
def shuffle_data(data):
    return random.sample(data, len(data))

# Splits the dataset into two groups based on the 'label' field
def partition_data(data):
    return (
        [entry for entry in data if entry['label'] == 1],
        [entry for entry in data if entry['label'] == 0]
    )

# Combines shuffling and partitioning, then increments the epoch counter
def update_epoch(data, epoch):
    return (partition_data(shuffle_data(data)), epoch + 1)

# Displays the current system environment along with additional information
def show_environment(info):
    return f"Current environment: {dict(os.environ)}\n{info}"

# Attempts to install a specified package using pip
def add_package(package):
    try:
        result = run(["pip", "install", package], text=True, capture_output=True, check=True)
        return result.stdout
    except CalledProcessError as e:
        return e.stderr

# Executes a shell command and returns the output and error messages
def run_shell_command(command):
    if not command:
        return "", "Invalid or empty command."
    result = run(command, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

# Initializes and returns a list of available tools
def setup_tools():
    return [
        Tool(name="EnvChecker", func=show_environment, description="Displays the current environment configuration."),
        Tool(name="PackageManager", func=add_package, description="Handles the installation of required packages.")
    ]

# Logs the execution of a command and its outcome
def execute_with_logging(command):
    logger.info(f"Running command: {command}")
    output, error = run_shell_command(command)
    if error:
        logger.error(f"Command failed: {error}")
    else:
        logger.success(f"Command succeeded: {output}")
    return not error

# Retries a command execution up to a specified number of attempts
def retry_command(agent, command, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Maximum retry attempts reached. Aborting.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if execute_with_logging(command):
        logger.success("Command executed successfully!")
        return
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing missing packages.")
    time.sleep(5)
    retry_command(agent, command, attempt + 1, max_attempts)

# Executes a command using an agent with retry logic
def run_with_agent(command, max_attempts):
    tools = setup_tools()
    agent = create_agent(tools=tools)
    retry_command(agent, command, 0, max_attempts)

# Measures the execution time of a function
def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return wrapper

# Initiates the process with timing and logging
@time_function
def begin_process(command, max_attempts):
    logger.info("Process initiation...")
    run_with_agent(command, max_attempts)

# Logs the command execution to a file
def record_command(command):
    try:
        with open("command_log.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
    except IOError as e:
        logger.error(f"Failed to log command: {e}")

# Counts down from a specified number of seconds
def countdown(seconds):
    if seconds > 0:
        logger.info(f"Countdown: {seconds} seconds remaining")
        time.sleep(1)
        countdown(seconds - 1)
    else:
        logger.success("Countdown finished!")

# Executes a command with optional countdown and retry logic
def execute(command, max_attempts=5, countdown_time=0):
    record_command(command)
    if countdown_time > 0:
        countdown(countdown_time)
    begin_process(command, max_attempts)

if __name__ == "__main__":
    click.command()(execute)()