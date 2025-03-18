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
    Shuffles a list of items and categorizes them based on their 'label' attribute.
    
    Args:
        items (list): A list of dictionaries, each containing a 'label' key.
        
    Returns:
        tuple: Two lists, one containing items with label 1 (positive), 
               and the other containing items with label 0 (negative).
    """
    shuffled = random.sample(items, len(items))
    positive = [item for item in shuffled if item['label'] == 1]
    negative = [item for item in shuffled if item['label'] == 0]
    return positive, negative

def process_items(items, step):
    """
    Processes a list of items by shuffling and categorizing them, and increments the step counter.
    
    Args:
        items (list): A list of items to process.
        step (int): The current step counter.
        
    Returns:
        tuple: A tuple containing the shuffled and categorized items, and the incremented step counter.
    """
    return shuffle_and_categorize(items), step + 1

def show_environment(info):
    """
    Displays the current system environment details along with additional information.
    
    Args:
        info (str): Additional information to display.
        
    Returns:
        str: A string containing the system environment details and the additional information.
    """
    return f"System environment details: {dict(os.environ)}\n{info}"

def install_pkg(pkg):
    """
    Installs a Python package using pip.
    
    Args:
        pkg (str): The name of the package to install.
        
    Returns:
        str: The output of the pip install command.
    """
    result = run(["pip", "install", pkg], text=True, capture_output=True, check=True)
    return result.stdout

def run_command(cmd):
    """
    Executes a shell command and captures its output and error.
    
    Args:
        cmd (str): The command to execute.
        
    Returns:
        tuple: A tuple containing the command's stdout and stderr.
    """
    if not cmd:
        return "", "No valid command provided."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def setup_tools():
    """
    Sets up and returns a list of tools that can be used by an agent.
    
    Returns:
        list: A list of Tool objects, each representing a specific functionality.
    """
    return [
        Tool(name="EnvViewer", func=show_environment, description="Displays the current system environment."),
        Tool(name="PackageManager", func=install_pkg, description="Manages package installations.")
    ]

def log_cmd_execution(cmd):
    """
    Logs the execution of a command and its output.
    
    Args:
        cmd (str): The command to execute and log.
        
    Returns:
        bool: True if the command executed successfully, False otherwise.
    """
    logger.info(f"Executing command: {cmd}")
    output, error = run_command(cmd)
    if error:
        logger.error(f"Execution failed: {error}")
        return False
    logger.success(f"Execution successful: {output}")
    return True

def retry_cmd(agent, cmd, attempt, max_attempts):
    """
    Retries a command execution up to a maximum number of attempts.
    
    Args:
        agent: The agent responsible for executing the command.
        cmd (str): The command to execute.
        attempt (int): The current attempt number.
        max_attempts (int): The maximum number of retry attempts.
    """
    if attempt >= max_attempts:
        logger.error("Max retry attempts reached. Stopping.")
        return
    logger.info(f"Retry attempt: {attempt + 1}/{max_attempts}")
    if log_cmd_execution(cmd):
        logger.success("Command completed successfully!")
    else:
        agent.run("Check environment variables and dependencies.")
        agent.run("Fix environment issues by installing missing packages.")
        time.sleep(5)
        retry_cmd(agent, cmd, attempt + 1, max_attempts)

def execute_with_retry(cmd, max_attempts):
    """
    Executes a command with retry logic using an agent.
    
    Args:
        cmd (str): The command to execute.
        max_attempts (int): The maximum number of retry attempts.
    """
    agent = create_agent(tools=setup_tools())
    retry_cmd(agent, cmd, 0, max_attempts)

def time_function(func):
    """
    A decorator that measures and logs the execution time of a function.
    
    Args:
        func: The function to time.
        
    Returns:
        function: The wrapped function with timing logic.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_function
def start_process(cmd, max_attempts):
    """
    Initiates the process of executing a command with retry logic.
    
    Args:
        cmd (str): The command to execute.
        max_attempts (int): The maximum number of retry attempts.
    """
    logger.info("Initiating process...")
    execute_with_retry(cmd, max_attempts)

def log_cmd(cmd):
    """
    Logs a command to a file for record-keeping.
    
    Args:
        cmd (str): The command to log.
    """
    try:
        with open("command_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except Exception as e:
        logger.error(f"Logging command failed: {e}")

def countdown(seconds):
    """
    Initiates a countdown timer and logs the remaining time.
    
    Args:
        seconds (int): The duration of the countdown in seconds.
    """
    if seconds > 0:
        logger.info(f"Time left: {seconds} seconds")
        time.sleep(1)
        countdown(seconds - 1)
    else:
        logger.success("Countdown complete!")

def execute_with_settings(cmd, max_attempts=5, timer_duration=0):
    """
    Executes a command with specified settings, including retry attempts and a countdown timer.
    
    Args:
        cmd (str): The command to execute.
        max_attempts (int): The maximum number of retry attempts.
        timer_duration (int): The duration of the countdown timer before execution.
    """
    log_cmd(cmd)
    if timer_duration > 0:
        countdown(timer_duration)
    start_process(cmd, max_attempts)

def backup_logs():
    """
    Backs up the command log file with a timestamped filename.
    """
    try:
        with open("command_log.txt", "r") as log_file:
            logs = log_file.read()
        with open(f"command_log_backup_{time.strftime('%Y%m%d%H%M%S')}.txt", "w") as backup_file:
            backup_file.write(logs)
        logger.success("Logs backed up successfully!")
    except Exception as e:
        logger.error(f"Backup failed: {e}")

if __name__ == "__main__":
    typer.run(execute_with_settings)
    backup_logs()