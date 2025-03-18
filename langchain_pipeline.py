import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_and_separate(items):
    randomized = random.sample(items, len(items))
    return [i for i in randomized if i['label'] == 1], [i for i in randomized if i['label'] == 0]

def handle_data(items, counter):
    return randomize_and_separate(items), counter + 1

def display_system_info(details):
    return f"System environment details: {dict(os.environ)}\n{details}"

def install_dependency(dependency):
    output = run(["pip", "install", dependency], text=True, capture_output=True, check=True)
    return output.stdout

def execute_command(command):
    if not command:
        return "", "No valid command provided."
    output = run(command, shell=True, text=True, capture_output=True)
    return output.stdout, output.stderr

def initialize_tools():
    return [
        Tool(name="EnvViewer", func=display_system_info, description="Displays the current system environment."),
        Tool(name="PackageManager", func=install_dependency, description="Manages package installations.")
    ]

def log_command(command):
    logger.info(f"Executing command: {command}")
    stdout, stderr = execute_command(command)
    if stderr:
        logger.error(f"Execution failed: {stderr}")
        return False
    logger.success(f"Execution successful: {stdout}")
    return True

def retry_command(agent, command, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Max retry attempts reached. Stopping.")
        return
    logger.info(f"Retry attempt: {attempt + 1}/{max_attempts}")
    if log_command(command):
        logger.success("Command completed successfully!")
    else:
        agent.run("Check environment variables and dependencies.")
        agent.run("Fix environment issues by installing missing packages.")
        time.sleep(5)
        retry_command(agent, command, attempt + 1, max_attempts)

def execute_with_retries(command, max_attempts):
    agent = create_agent(tools=initialize_tools())
    retry_command(agent, command, 0, max_attempts)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@measure_time
def begin_process(command, max_attempts):
    logger.info("Starting process...")
    execute_with_retries(command, max_attempts)

def log_command_to_file(command):
    try:
        with open("command_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
    except Exception as e:
        logger.error(f"Logging command failed: {e}")

def countdown(duration):
    if duration > 0:
        logger.info(f"Time left: {duration} seconds")
        time.sleep(1)
        countdown(duration - 1)
    else:
        logger.success("Countdown complete!")

def execute_with_config(command, max_attempts=5, countdown_duration=0):
    log_command_to_file(command)
    if countdown_duration > 0:
        countdown(countdown_duration)
    begin_process(command, max_attempts)

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