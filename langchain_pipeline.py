import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_and_separate(data):
    randomized = random.sample(data, len(data))
    return [item for item in randomized if item['label'] == 1], [item for item in randomized if item['label'] == 0]

def handle_data(data, counter):
    return randomize_and_separate(data), counter + 1

def display_system_info(additional_info):
    return f"System environment details: {dict(os.environ)}\n{additional_info}"

def install_package(package_name):
    result = run(["pip", "install", package_name], text=True, capture_output=True, check=True)
    return result.stdout

def execute_shell_command(command):
    if not command:
        return "", "No valid command provided."
    result = run(command, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools():
    return [
        Tool(name="EnvViewer", func=display_system_info, description="Displays the current system environment."),
        Tool(name="PackageManager", func=install_package, description="Manages package installations.")
    ]

def log_command_execution(command):
    logger.info(f"Executing command: {command}")
    output, error = execute_shell_command(command)
    if error:
        logger.error(f"Execution failed: {error}")
        return False
    logger.success(f"Execution successful: {output}")
    return True

def retry_command_execution(agent, command, current_attempt, max_attempts):
    if current_attempt >= max_attempts:
        logger.error("Max retry attempts reached. Stopping.")
        return
    logger.info(f"Retry attempt: {current_attempt + 1}/{max_attempts}")
    if log_command_execution(command):
        logger.success("Command completed successfully!")
    else:
        agent.run("Check environment variables and dependencies.")
        agent.run("Fix environment issues by installing missing packages.")
        time.sleep(5)
        retry_command_execution(agent, command, current_attempt + 1, max_attempts)

def execute_with_retries(command, max_attempts):
    agent = create_agent(tools=initialize_tools())
    retry_command_execution(agent, command, 0, max_attempts)

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@measure_execution_time
def initiate_process(command, max_attempts):
    logger.info("Initiating process...")
    execute_with_retries(command, max_attempts)

def log_command_to_file(command):
    try:
        with open("command_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
    except Exception as e:
        logger.error(f"Logging command failed: {e}")

def start_countdown(duration):
    if duration > 0:
        logger.info(f"Time left: {duration} seconds")
        time.sleep(1)
        start_countdown(duration - 1)
    else:
        logger.success("Countdown complete!")

def execute_with_config(command, max_attempts=5, countdown_duration=0):
    log_command_to_file(command)
    if countdown_duration > 0:
        start_countdown(countdown_duration)
    initiate_process(command, max_attempts)

def backup_command_logs():
    try:
        with open("command_log.txt", "r") as log_file:
            logs = log_file.read()
        with open(f"command_log_backup_{time.strftime('%Y%m%d%H%M%S')}.txt", "w") as backup_file:
            backup_file.write(logs)
        logger.success("Logs backed up successfully!")
    except Exception as e:
        logger.error(f"Backup failed: {e}")

if __name__ == "__main__":
    typer.run(execute_with_config)
    backup_command_logs()