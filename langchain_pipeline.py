import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def shuffle_and_split(items):
    randomized = random.sample(items, len(items))
    return [i for i in randomized if i['label'] == 1], [i for i in randomized if i['label'] == 0]

def process_items(items, counter):
    return shuffle_and_split(items), counter + 1

def get_system_details(details):
    return f"System environment details: {dict(os.environ)}\n{details}"

def install_package(package):
    output = run(["pip", "install", package], text=True, capture_output=True, check=True)
    return output.stdout

def run_shell_command(command):
    if not command:
        return "", "No valid command provided."
    output = run(command, shell=True, text=True, capture_output=True)
    return output.stdout, output.stderr

def setup_tools():
    return [
        Tool(name="EnvViewer", func=get_system_details, description="Shows system environment details."),
        Tool(name="PackageManager", func=install_package, description="Handles package installations.")
    ]

def log_and_execute(command):
    logger.info(f"Running command: {command}")
    stdout, stderr = run_shell_command(command)
    if stderr:
        logger.error(f"Command failed: {stderr}")
        return False
    logger.success(f"Command succeeded: {stdout}")
    return True

def retry_execution(agent, command, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Maximum retries reached. Aborting.")
        return
    logger.info(f"Attempt {attempt + 1} of {max_attempts}")
    if log_and_execute(command):
        logger.success("Command executed successfully!")
    else:
        agent.run("Verify environment variables and dependencies.")
        agent.run("Resolve issues by installing required packages.")
        time.sleep(5)
        retry_execution(agent, command, attempt + 1, max_attempts)

def execute_with_retry(command, max_attempts):
    agent = create_agent(tools=setup_tools())
    retry_execution(agent, command, 0, max_attempts)

def time_execution(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_execution
def start_process(command, max_attempts):
    logger.info("Initiating process...")
    execute_with_retry(command, max_attempts)

def log_command_history(command):
    try:
        with open("command_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
    except Exception as e:
        logger.error(f"Failed to log command: {e}")

def display_countdown(duration):
    if duration > 0:
        logger.info(f"Remaining time: {duration} seconds")
        time.sleep(1)
        display_countdown(duration - 1)
    else:
        logger.success("Countdown finished!")

def execute_with_settings(command, max_attempts=5, countdown_duration=0):
    log_command_history(command)
    if countdown_duration > 0:
        display_countdown(countdown_duration)
    start_process(command, max_attempts)

def backup_logs():
    try:
        with open("command_log.txt", "r") as log_file:
            logs = log_file.read()
        with open(f"command_log_backup_{time.strftime('%Y%m%d%H%M%S')}.txt", "w") as backup_file:
            backup_file.write(logs)
        logger.success("Backup completed successfully!")
    except Exception as e:
        logger.error(f"Backup error: {e}")

if __name__ == "__main__":
    typer.run(execute_with_settings)
    backup_logs()