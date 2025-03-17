import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_and_split(items):
    """
    Randomizes the order of items and splits them into two lists based on their 'label' value.
    
    Args:
        items (list): A list of dictionaries, each containing a 'label' key.
        
    Returns:
        tuple: Two lists, one containing items with label == 1, the other with label == 0.
    """
    randomized = random.sample(items, len(items))
    return [item for item in randomized if item['label'] == 1], [item for item in randomized if item['label'] == 0]

def process_and_update(items, step):
    """
    Processes the items by randomizing and splitting them, and increments the step counter.
    
    Args:
        items (list): A list of items to process.
        step (int): The current step counter.
        
    Returns:
        tuple: A tuple containing the randomized and split items, and the incremented step counter.
    """
    return randomize_and_split(items), step + 1

def display_environment(info):
    """
    Displays the current environment settings along with additional information.
    
    Args:
        info (str): Additional information to display.
        
    Returns:
        str: A string containing environment details and the provided info.
    """
    return f"Environment details: {dict(os.environ)}\n{info}"

def install_package(package):
    """
    Installs a Python package using pip.
    
    Args:
        package (str): The name of the package to install.
        
    Returns:
        str: The output of the pip install command.
    """
    return run(["pip", "install", package], text=True, capture_output=True, check=True).stdout

def execute_command(cmd):
    """
    Executes a shell command and returns the output and error messages.
    
    Args:
        cmd (str): The command to execute.
        
    Returns:
        tuple: A tuple containing the command's stdout and stderr.
    """
    if not cmd:
        return "", "Command is empty or invalid."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools():
    """
    Initializes and returns a list of available tools.
    
    Returns:
        list: A list of Tool objects, each representing a specific functionality.
    """
    return [
        Tool(name="EnvViewer", func=display_environment, description="Displays the current environment settings."),
        Tool(name="PackageManager", func=install_package, description="Installs required packages.")
    ]

def log_execution(cmd):
    """
    Logs the execution of a command and handles its output.
    
    Args:
        cmd (str): The command to execute and log.
        
    Returns:
        bool: True if the command succeeded, False otherwise.
    """
    logger.info(f"Running command: {cmd}")
    output, error = execute_command(cmd)
    if error:
        logger.error(f"Command failed: {error}")
        return False
    logger.success(f"Command succeeded: {output}")
    return True

def retry_execution(agent, cmd, attempt, max_attempts):
    """
    Retries the execution of a command up to a maximum number of attempts.
    
    Args:
        agent: The agent responsible for handling retries.
        cmd (str): The command to execute.
        attempt (int): The current attempt number.
        max_attempts (int): The maximum number of retry attempts.
    """
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
    """
    Executes a command with retries using an agent.
    
    Args:
        cmd (str): The command to execute.
        max_attempts (int): The maximum number of retry attempts.
    """
    agent = create_agent(tools=initialize_tools())
    return retry_execution(agent, cmd, 0, max_attempts)

def time_execution(func):
    """
    A decorator that logs the execution time of a function.
    
    Args:
        func: The function to be timed.
        
    Returns:
        function: The wrapped function with timing functionality.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_execution
def begin_process(cmd, max_attempts):
    """
    Begins the process of executing a command with retries.
    
    Args:
        cmd (str): The command to execute.
        max_attempts (int): The maximum number of retry attempts.
    """
    logger.info("Starting process...")
    return execute_with_retries(cmd, max_attempts)

def log_command(cmd):
    """
    Logs a command to a file for record-keeping.
    
    Args:
        cmd (str): The command to log.
    """
    try:
        with open("command_log.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except IOError as e:
        logger.error(f"Failed to log command: {e}")

def start_countdown(seconds):
    """
    Starts a countdown timer for the specified number of seconds.
    
    Args:
        seconds (int): The duration of the countdown in seconds.
    """
    if seconds > 0:
        logger.info(f"Countdown: {seconds} seconds remaining")
        time.sleep(1)
        start_countdown(seconds - 1)
    else:
        logger.success("Countdown finished!")

def execute_with_config(cmd, max_attempts=5, timer_duration=0):
    """
    Executes a command with a configurable number of retries and an optional countdown timer.
    
    Args:
        cmd (str): The command to execute.
        max_attempts (int): The maximum number of retry attempts (default is 5).
        timer_duration (int): The duration of the countdown timer in seconds (default is 0).
    """
    log_command(cmd)
    if timer_duration > 0:
        start_countdown(timer_duration)
    begin_process(cmd, max_attempts)

if __name__ == "__main__":
    typer.run(execute_with_config)