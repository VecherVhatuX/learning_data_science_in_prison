import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def shuffle_and_categorize(items):
    """
    Randomizes the order of items and sorts them based on their 'label' attribute.
    
    Args:
        items (list): A list of dictionaries, each containing a 'label' key.
        
    Returns:
        tuple: Two lists, one containing items with label == 1, the other with label == 0.
    """
    shuffled = random.sample(items, len(items))
    return (
        [item for item in shuffled if item['label'] == 1],
        [item for item in shuffled if item['label'] == 0]
    )

def process_data(items, iteration):
    """
    Handles data by randomizing and sorting it, and increments the iteration counter.
    
    Args:
        items (list): A list of items to be processed.
        iteration (int): The current iteration count.
        
    Returns:
        tuple: A tuple containing the randomized and sorted items, and the incremented iteration count.
    """
    return (shuffle_and_categorize(items), iteration + 1)

def show_environment(info):
    """
    Displays the current environment settings along with additional information.
    
    Args:
        info (str): Additional information to display alongside environment details.
        
    Returns:
        str: A string containing environment details and the provided information.
    """
    return f"Environment details: {dict(os.environ)}\n{info}"

def add_package(package):
    """
    Installs a Python package using pip.
    
    Args:
        package (str): The name of the package to install.
        
    Returns:
        str: The output of the pip install command, or an error message if the installation fails.
    """
    result = run(["pip", "install", package], text=True, capture_output=True, check=True)
    return result.stdout if not result.returncode else result.stderr

def run_shell_command(cmd):
    """
    Executes a shell command and captures its output and error.
    
    Args:
        cmd (str): The shell command to execute.
        
    Returns:
        tuple: A tuple containing the command's stdout and stderr.
    """
    if not cmd:
        return "", "Command is empty or invalid."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def setup_tools():
    """
    Initializes and returns a list of tools available for use.
    
    Returns:
        list: A list of Tool objects, each representing a specific functionality.
    """
    return [
        Tool(name="EnvViewer", func=show_environment, description="Displays the current environment settings."),
        Tool(name="PackageManager", func=add_package, description="Installs required packages.")
    ]

def log_shell_command(cmd):
    """
    Logs the execution of a shell command and its output.
    
    Args:
        cmd (str): The shell command to execute and log.
        
    Returns:
        bool: True if the command succeeds, False otherwise.
    """
    logger.info(f"Running command: {cmd}")
    output, error = run_shell_command(cmd)
    if error:
        logger.error(f"Command failed: {error}")
        return False
    logger.success(f"Command succeeded: {output}")
    return True

def retry_shell_command(agent, cmd, attempt, max_attempts):
    """
    Retries a shell command up to a maximum number of attempts.
    
    Args:
        agent: The agent responsible for handling the command.
        cmd (str): The shell command to retry.
        attempt (int): The current attempt number.
        max_attempts (int): The maximum number of retry attempts.
    """
    if attempt >= max_attempts:
        logger.error("Maximum retries reached. Aborting.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if log_shell_command(cmd):
        logger.success("Command executed successfully!")
        return
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing missing packages.")
    time.sleep(5)
    retry_shell_command(agent, cmd, attempt + 1, max_attempts)

def execute_with_retries(cmd, max_attempts):
    """
    Executes a shell command with retry logic.
    
    Args:
        cmd (str): The shell command to execute.
        max_attempts (int): The maximum number of retry attempts.
    """
    tools = setup_tools()
    agent = create_agent(tools=tools)
    retry_shell_command(agent, cmd, 0, max_attempts)

def measure_execution_time(func):
    """
    A decorator that logs the execution time of a function.
    
    Args:
        func: The function to be timed.
        
    Returns:
        function: The wrapped function with timing logic.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@measure_execution_time
def start_execution(cmd, max_attempts):
    """
    Begins the process of executing a shell command with retries.
    
    Args:
        cmd (str): The shell command to execute.
        max_attempts (int): The maximum number of retry attempts.
    """
    logger.info("Starting process...")
    execute_with_retries(cmd, max_attempts)

def record_command(cmd):
    """
    Logs the executed command to a history file.
    
    Args:
        cmd (str): The command to log.
    """
    try:
        with open("command_history.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except IOError as e:
        logger.error(f"Failed to log command: {e}")

def countdown_timer(seconds):
    """
    Starts a countdown timer and logs the remaining time.
    
    Args:
        seconds (int): The duration of the timer in seconds.
    """
    if seconds > 0:
        logger.info(f"Timer: {seconds} seconds remaining")
        time.sleep(1)
        countdown_timer(seconds - 1)
    else:
        logger.success("Timer finished!")

def execute_command(cmd: str, max_attempts: int = 5, timer_duration: int = 0):
    """
    Runs a shell command with optional retries and a timer.
    
    Args:
        cmd (str): The shell command to execute.
        max_attempts (int): The maximum number of retry attempts.
        timer_duration (int): The duration of the timer in seconds.
    """
    record_command(cmd)
    if timer_duration > 0:
        countdown_timer(timer_duration)
    start_execution(cmd, max_attempts)

if __name__ == "__main__":
    typer.run(execute_command)