import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def shuffle_and_split(items):
    """
    Shuffles the list of items and splits them into two lists based on their 'label' value.
    
    Args:
        items (list): A list of dictionaries, each containing a 'label' key.
    
    Returns:
        tuple: Two lists, one containing items with label 1, the other with label 0.
    """
    randomized = random.sample(items, len(items))
    return [i for i in randomized if i['label'] == 1], [i for i in randomized if i['label'] == 0]

def process_items(items, counter):
    """
    Processes items by shuffling and splitting them, and increments the counter.
    
    Args:
        items (list): A list of items to process.
        counter (int): A counter to increment.
    
    Returns:
        tuple: A tuple containing the shuffled and split items, and the incremented counter.
    """
    return shuffle_and_split(items), counter + 1

def get_system_details(info):
    """
    Returns a string containing system environment details and additional info.
    
    Args:
        info (str): Additional information to include in the output.
    
    Returns:
        str: A string containing system environment details and the provided info.
    """
    return f"System environment: {dict(os.environ)}\n{info}"

def install_package(pkg):
    """
    Installs a Python package using pip.
    
    Args:
        pkg (str): The name of the package to install.
    
    Returns:
        str: The output of the pip install command.
    """
    output = run(["pip", "install", pkg], text=True, capture_output=True, check=True)
    return output.stdout

def run_shell_command(cmd):
    """
    Executes a shell command and returns the output.
    
    Args:
        cmd (str): The shell command to execute.
    
    Returns:
        tuple: A tuple containing the stdout and stderr of the command.
    """
    if not cmd:
        return "", "No command provided."
    output = run(cmd, shell=True, text=True, capture_output=True)
    return output.stdout, output.stderr

def setup_tools():
    """
    Sets up and returns a list of tools for the agent to use.
    
    Returns:
        list: A list of Tool objects.
    """
    return [
        Tool(name="EnvInspector", func=get_system_details, description="Shows system environment details."),
        Tool(name="PkgInstaller", func=install_package, description="Handles package installations.")
    ]

def log_execution(cmd):
    """
    Logs and executes a shell command, handling errors if they occur.
    
    Args:
        cmd (str): The shell command to execute.
    
    Returns:
        bool: True if the command executed successfully, False otherwise.
    """
    logger.info(f"Running command: {cmd}")
    stdout, stderr = run_shell_command(cmd)
    if stderr:
        logger.error(f"Command error: {stderr}")
        return False
    logger.success(f"Command output: {stdout}")
    return True

def retry_execution(agent, cmd, attempt, max_attempts):
    """
    Retries executing a shell command if it fails, up to a maximum number of attempts.
    
    Args:
        agent: The agent responsible for handling retries.
        cmd (str): The shell command to execute.
        attempt (int): The current attempt number.
        max_attempts (int): The maximum number of retry attempts.
    """
    if attempt >= max_attempts:
        logger.error("Max retries exceeded. Stopping.")
        return
    logger.info(f"Retry {attempt + 1} of {max_attempts}")
    if log_execution(cmd):
        logger.success("Command successful!")
    else:
        agent.run("Check environment variables and dependencies.")
        agent.run("Install missing packages if needed.")
        time.sleep(5)
        retry_execution(agent, cmd, attempt + 1, max_attempts)

def execute_with_retry(cmd, max_attempts):
    """
    Executes a shell command with retry logic.
    
    Args:
        cmd (str): The shell command to execute.
        max_attempts (int): The maximum number of retry attempts.
    """
    agent = create_agent(tools=setup_tools())
    retry_execution(agent, cmd, 0, max_attempts)

def time_execution(func):
    """
    A decorator that logs the execution time of a function.
    
    Args:
        func: The function to be timed.
    
    Returns:
        function: The wrapped function with timing logic.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_execution
def start_process(cmd, max_attempts):
    """
    Initiates a process with retry logic and logs the start and end of the process.
    
    Args:
        cmd (str): The shell command to execute.
        max_attempts (int): The maximum number of retry attempts.
    """
    logger.info("Process initiated...")
    execute_with_retry(cmd, max_attempts)

def log_command(cmd):
    """
    Logs a command to a file with a timestamp.
    
    Args:
        cmd (str): The command to log.
    """
    try:
        with open("command_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except Exception as e:
        logger.error(f"Logging failed: {e}")

def display_countdown(duration):
    """
    Displays a countdown in the logs for the specified duration.
    
    Args:
        duration (int): The duration of the countdown in seconds.
    """
    if duration > 0:
        logger.info(f"Remaining time: {duration} seconds")
        time.sleep(1)
        display_countdown(duration - 1)
    else:
        logger.success("Countdown finished!")

def execute_with_settings(cmd, max_attempts=5, countdown_duration=0):
    """
    Executes a command with specified settings, including retry attempts and a countdown.
    
    Args:
        cmd (str): The shell command to execute.
        max_attempts (int): The maximum number of retry attempts.
        countdown_duration (int): The duration of the countdown before execution.
    """
    log_command(cmd)
    if countdown_duration > 0:
        display_countdown(countdown_duration)
    start_process(cmd, max_attempts)

def backup_logs():
    """
    Backs up the command log to a new file with a timestamp.
    """
    try:
        with open("command_log.txt", "r") as log_file:
            logs = log_file.read()
        with open(f"command_log_backup_{time.strftime('%Y%m%d%H%M%S')}.txt", "w") as backup_file:
            backup_file.write(logs)
        logger.success("Backup completed!")
    except Exception as e:
        logger.error(f"Backup error: {e}")

if __name__ == "__main__":
    typer.run(execute_with_settings)
    backup_logs()