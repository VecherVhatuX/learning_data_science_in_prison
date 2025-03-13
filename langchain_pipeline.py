import os
import time
import random
from subprocess import run, CalledProcessError
import typer
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger

# Randomizes the order of samples in the list
def randomize_samples(samples: List[Dict]) -> List[Dict]:
    return random.sample(samples, len(samples))

# Splits samples into two lists based on the 'label' key (1 or 0)
def split_samples(samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    return (
        [item for item in samples if item['label'] == 1],
        [item for item in samples if item['label'] == 0]
    )

# Increments the epoch and returns randomized and split samples
def increment_epoch(samples: List[Dict], epoch: int) -> Tuple[Tuple[List[Dict], List[Dict]], int]:
    return split_samples(randomize_samples(samples)), epoch + 1

# Displays the current environment configurations along with additional data
def display_environment(data: str) -> str:
    return f"Current configurations: {dict(os.environ)}\n{data}"

# Attempts to install a package using pip and returns the output or error
def install_package(data: str) -> str:
    try:
        return run(
            ["pip", "install", data],
            text=True,
            capture_output=True,
            check=True
        ).stdout
    except CalledProcessError as error:
        return error.stderr

# Executes a shell command and returns the stdout and stderr
def execute_command(cmd: str) -> Tuple[str, str]:
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

# Initializes a list of tools with their respective functions and descriptions
def initialize_tools() -> List[Tool]:
    return [
        Tool(name="EnvironmentInspector", func=display_environment, description="Shows current configuration details."),
        Tool(name="DependencyInstaller", func=install_package, description="Installs necessary packages.")
    ]

# Executes a command and logs the output or error
def run_command_with_feedback(cmd: str) -> bool:
    logger.info(f"Executing command: {cmd}")
    output, error = execute_command(cmd)
    if error:
        logger.error(f"Execution failed: {error}")
        return False
    logger.success(f"Command executed successfully: {output}")
    return True

# Attempts to execute a command with retries, using an agent to resolve issues
def attempt_command(agent, cmd: str, attempt: int, max_retries: int) -> bool:
    if attempt >= max_retries:
        logger.error("Reached maximum retries. Aborting.")
        return False
    logger.info(f"Attempt number: {attempt + 1}/{max_retries}")
    if run_command_with_feedback(cmd):
        logger.success("Command executed successfully!")
        return True
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing required packages.")
    time.sleep(5)
    return attempt_command(agent, cmd, attempt + 1, max_retries)

# Executes a command using an agent with a specified number of retries
def execute_with_agent(cmd: str, max_retries: int):
    tools = initialize_tools()
    agent = create_agent(tools=tools)
    attempt_command(agent, cmd, 0, max_retries)

# Decorator to track the execution time of a function
def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Duration: {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

# Starts the process, logs the command, and executes it with an agent
@time_tracker
def start_process(cmd: str, max_retries: int):
    logger.info("Process is starting...")
    execute_with_agent(cmd, max_retries)

# Logs the command to a file with a timestamp
def log_command(cmd: str):
    with open("command_log.txt", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")

# A countdown timer that logs the remaining time
def timer(seconds: int):
    def timer_helper(sec):
        if sec > 0:
            logger.info(f"Timer: {sec} seconds remaining")
            time.sleep(1)
            timer_helper(sec - 1)
        else:
            logger.success("Timer finished!")

    timer_helper(seconds)

# Main function to log the command, start a timer (if specified), and execute the process
def main(cmd: str, max_retries: int = 5, countdown_time: int = 0):
    log_command(cmd)
    if countdown_time > 0:
        timer(countdown_time)
    start_process(cmd, max_retries)

if __name__ == "__main__":
    typer.run(main)