import os
import time
import random
from subprocess import run, CalledProcessError
import click
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import reduce

def shuffle_data(data):
    """Shuffle the given data randomly."""
    return random.sample(data, len(data))

def partition_data(data):
    """Partition the data into two lists based on the 'label' key (1 or 0)."""
    return (
        [entry for entry in data if entry['label'] == 1],
        [entry for entry in data if entry['label'] == 0]
    )

def update_epoch(data, epoch):
    """Shuffle and partition the data, then increment the epoch."""
    return (partition_data(shuffle_data(data)), epoch + 1)

def show_environment(info):
    """Display the current environment variables along with additional info."""
    return f"Current environment: {dict(os.environ)}\n{info}"

def add_package(package):
    """Install a Python package using pip."""
    try:
        result = run(["pip", "install", package], text=True, capture_output=True, check=True)
        return result.stdout
    except CalledProcessError as e:
        return e.stderr

def run_shell_command(command):
    """Execute a shell command and return the output and error."""
    if not command:
        return "", "Invalid or empty command."
    result = run(command, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def setup_tools():
    """Set up and return a list of tools available for the agent."""
    return [
        Tool(name="EnvChecker", func=show_environment, description="Shows current environment settings."),
        Tool(name="PackageManager", func=add_package, description="Installs necessary packages.")
    ]

def execute_with_logging(command):
    """Execute a command with logging for success or failure."""
    logger.info(f"Running command: {command}")
    output, error = run_shell_command(command)
    if error:
        logger.error(f"Command failed: {error}")
    else:
        logger.success(f"Command succeeded: {output}")
    return not error

def retry_command(agent, command, attempt, max_attempts):
    """Retry a command with a specified number of attempts, using the agent to fix issues."""
    if attempt >= max_attempts:
        logger.error("Max retries reached. Stopping.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if execute_with_logging(command):
        logger.success("Command executed!")
        return
    agent.run("Check environment variables and dependencies.")
    agent.run("Try to fix environment issues by installing missing packages.")
    time.sleep(5)
    retry_command(agent, command, attempt + 1, max_attempts)

def run_with_agent(command, max_attempts):
    """Run a command with the help of an agent, retrying if necessary."""
    tools = setup_tools()
    agent = create_agent(tools=tools)
    retry_command(agent, command, 0, max_attempts)

def time_function(func):
    """Decorator to measure and log the execution time of a function."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_function
def begin_process(command, max_attempts):
    """Start the process of executing a command with retries."""
    logger.info("Starting process...")
    run_with_agent(command, max_attempts)

def record_command(command):
    """Log the command to a file with a timestamp."""
    try:
        with open("command_log.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
    except IOError as e:
        logger.error(f"Logging failed: {e}")

def countdown(seconds):
    """Perform a countdown for the specified number of seconds."""
    if seconds > 0:
        logger.info(f"Countdown: {seconds} seconds left")
        time.sleep(1)
        countdown(seconds - 1)
    else:
        logger.success("Countdown complete!")

def execute(command, max_attempts=5, countdown_time=0):
    """Execute a command with optional countdown and retries."""
    record_command(command)
    if countdown_time > 0:
        countdown(countdown_time)
    begin_process(command, max_attempts)

if __name__ == "__main__":
    click.command()(execute)()