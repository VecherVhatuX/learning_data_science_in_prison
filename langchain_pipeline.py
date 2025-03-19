import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_and_separate(items):
    # Shuffle the items and split them based on their label value
    randomized = random.sample(items, len(items))
    return [i for i in randomized if i['label'] == 1], [i for i in randomized if i['label'] == 0]

def handle_data(items, counter):
    # Process the items and increment the counter
    return randomize_and_separate(items), counter + 1

def display_system_info(details):
    # Return a string with system environment details and additional info
    return f"System environment details: {dict(os.environ)}\n{details}"

def install_dependency(dependency):
    # Install a Python package using pip and return the output
    output = run(["pip", "install", dependency], text=True, capture_output=True, check=True)
    return output.stdout

def execute_command(command):
    # Execute a shell command and return the output and error messages
    if not command:
        return "", "No valid command provided."
    output = run(command, shell=True, text=True, capture_output=True)
    return output.stdout, output.stderr

def initialize_tools():
    # Create and return a list of available tools
    return [
        Tool(name="EnvViewer", func=display_system_info, description="Shows system environment details."),
        Tool(name="PackageManager", func=install_dependency, description="Handles package installations.")
    ]

def log_command(command):
    # Log the execution of a command and handle errors
    logger.info(f"Running command: {command}")
    stdout, stderr = execute_command(command)
    if stderr:
        logger.error(f"Command failed: {stderr}")
        return False
    logger.success(f"Command succeeded: {stdout}")
    return True

def retry_command(agent, command, attempt, max_attempts):
    # Retry a command if it fails, up to a maximum number of attempts
    if attempt >= max_attempts:
        logger.error("Maximum retries reached. Aborting.")
        return
    logger.info(f"Attempt {attempt + 1} of {max_attempts}")
    if log_command(command):
        logger.success("Command executed successfully!")
    else:
        agent.run("Verify environment variables and dependencies.")
        agent.run("Resolve issues by installing required packages.")
        time.sleep(5)
        retry_command(agent, command, attempt + 1, max_attempts)

def execute_with_retries(command, max_attempts):
    # Execute a command with retries using an agent
    agent = create_agent(tools=initialize_tools())
    retry_command(agent, command, 0, max_attempts)

def measure_time(func):
    # Measure the execution time of a function
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@measure_time
def begin_process(command, max_attempts):
    # Start the main process and execute the command with retries
    logger.info("Initiating process...")
    execute_with_retries(command, max_attempts)

def log_command_to_file(command):
    # Log the command to a file for future reference
    try:
        with open("command_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
    except Exception as e:
        logger.error(f"Failed to log command: {e}")

def countdown(duration):
    # Display a countdown timer for the specified duration
    if duration > 0:
        logger.info(f"Remaining time: {duration} seconds")
        time.sleep(1)
        countdown(duration - 1)
    else:
        logger.success("Countdown finished!")

def execute_with_config(command, max_attempts=5, countdown_duration=0):
    # Execute a command with configuration options
    log_command_to_file(command)
    if countdown_duration > 0:
        countdown(countdown_duration)
    begin_process(command, max_attempts)

def backup_command_logs():
    # Create a backup of the command log file
    try:
        with open("command_log.txt", "r") as log_file:
            logs = log_file.read()
        with open(f"command_log_backup_{time.strftime('%Y%m%d%H%M%S')}.txt", "w") as backup_file:
            backup_file.write(logs)
        logger.success("Backup completed successfully!")
    except Exception as e:
        logger.error(f"Backup error: {e}")

if __name__ == "__main__":
    typer.run(execute_with_config)
    backup_command_logs()