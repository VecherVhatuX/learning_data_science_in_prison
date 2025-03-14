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
    """
    Randomly shuffles the items in the list.
    
    Args:
        items (List): List of items to be randomized.
    
    Returns:
        List: A new list with items in random order.
    """
    return random.sample(items, len(items))

def separate_by_label(items):
    """
    Separates items into two lists based on their 'label' value.
    
    Args:
        items (List[Dict]): List of dictionaries containing a 'label' key.
    
    Returns:
        Tuple[List, List]: A tuple containing two lists, one for items with label 1 and another for label 0.
    """
    return (
        [item for item in items if item['label'] == 1],
        [item for item in items if item['label'] == 0]
    )

def process_epoch(items, epoch):
    """
    Processes a list of items by randomizing them and separating by label.
    
    Args:
        items (List): List of items to process.
        epoch (int): Current epoch number.
    
    Returns:
        Tuple[Tuple[List, List], int]: A tuple containing the separated lists and the incremented epoch number.
    """
    return (separate_by_label(randomize_items(items)), epoch + 1)

def display_env(info):
    """
    Displays the current environment settings along with additional info.
    
    Args:
        info (str): Additional information to display.
    
    Returns:
        str: A string containing environment details and the provided info.
    """
    return f"Environment details: {dict(os.environ)}\n{info}"

def install_package(pkg):
    """
    Installs a Python package using pip.
    
    Args:
        pkg (str): Name of the package to install.
    
    Returns:
        str: The output of the pip install command or the error message if it fails.
    """
    try:
        result = run(["pip", "install", pkg], text=True, capture_output=True, check=True)
        return result.stdout
    except CalledProcessError as e:
        return e.stderr

def execute_shell_cmd(cmd):
    """
    Executes a shell command and returns the output and error.
    
    Args:
        cmd (str): The shell command to execute.
    
    Returns:
        Tuple[str, str]: A tuple containing the command's stdout and stderr.
    """
    if not cmd:
        return "", "Command is empty or invalid."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools():
    """
    Initializes and returns a list of available tools.
    
    Returns:
        List[Tool]: A list of Tool objects with predefined functions and descriptions.
    """
    return [
        Tool(name="EnvViewer", func=display_env, description="Displays the current environment settings."),
        Tool(name="PackageManager", func=install_package, description="Installs required packages.")
    ]

def log_cmd_exec(cmd):
    """
    Logs the execution of a shell command and its output.
    
    Args:
        cmd (str): The shell command to execute and log.
    
    Returns:
        bool: True if the command succeeded, False otherwise.
    """
    logger.info(f"Running command: {cmd}")
    output, error = execute_shell_cmd(cmd)
    if error:
        logger.error(f"Command failed: {error}")
    else:
        logger.success(f"Command succeeded: {output}")
    return not error

def retry_cmd_exec(agent, cmd, attempt, max_attempts):
    """
    Attempts to execute a shell command with retries in case of failure.
    
    Args:
        agent: The agent responsible for resolving issues.
        cmd (str): The shell command to execute.
        attempt (int): Current attempt number.
        max_attempts (int): Maximum number of retry attempts.
    """
    if attempt >= max_attempts:
        logger.error("Maximum retries reached. Aborting.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if log_cmd_exec(cmd):
        logger.success("Command executed successfully!")
        return
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing missing packages.")
    time.sleep(5)
    retry_cmd_exec(agent, cmd, attempt + 1, max_attempts)

def execute_with_retries(cmd, max_attempts):
    """
    Executes a shell command with retries using an agent to resolve issues.
    
    Args:
        cmd (str): The shell command to execute.
        max_attempts (int): Maximum number of retry attempts.
    """
    tools = initialize_tools()
    agent = create_agent(tools=tools)
    retry_cmd_exec(agent, cmd, 0, max_attempts)

def time_execution(func):
    """
    Decorator to measure and log the execution time of a function.
    
    Args:
        func: The function to be timed.
    
    Returns:
        function: The wrapped function with timing logic.
    """
    @wraps(func)
    def timed_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return timed_func

@time_execution
def start_process(cmd, max_attempts):
    """
    Starts the process of executing a shell command with retries.
    
    Args:
        cmd (str): The shell command to execute.
        max_attempts (int): Maximum number of retry attempts.
    """
    logger.info("Starting process...")
    execute_with_retries(cmd, max_attempts)

def log_cmd(cmd):
    """
    Logs the executed command to a file.
    
    Args:
        cmd (str): The command to log.
    """
    try:
        with open("command_history.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except IOError as e:
        logger.error(f"Failed to log command: {e}")

def countdown(seconds):
    """
    Initiates a countdown timer and logs the remaining time.
    
    Args:
        seconds (int): The duration of the countdown in seconds.
    """
    if seconds > 0:
        logger.info(f"Timer: {seconds} seconds remaining")
        time.sleep(1)
        countdown(seconds - 1)
    else:
        logger.success("Timer finished!")

def run_cmd(cmd, max_attempts=5, timer_duration=0):
    """
    Executes a shell command with optional retries and a countdown timer.
    
    Args:
        cmd (str): The shell command to execute.
        max_attempts (int): Maximum number of retry attempts.
        timer_duration (int): Duration of the countdown timer before execution.
    """
    log_cmd(cmd)
    if timer_duration > 0:
        countdown(timer_duration)
    start_process(cmd, max_attempts)

if __name__ == "__main__":
    click.command()(run_cmd)()