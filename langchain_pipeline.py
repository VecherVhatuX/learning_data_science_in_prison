import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def shuffle_and_categorize(items):
    shuffled = random.sample(items, len(items))
    positive = [item for item in shuffled if item['label'] == 1]
    negative = [item for item in shuffled if item['label'] == 0]
    return positive, negative

def process_items(items, step):
    return shuffle_and_categorize(items), step + 1

def show_environment(info):
    return f"System environment details: {dict(os.environ)}\n{info}"

def install_pkg(pkg):
    result = run(["pip", "install", pkg], text=True, capture_output=True, check=True)
    return result.stdout

def run_command(cmd):
    if not cmd:
        return "", "No valid command provided."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def setup_tools():
    return [
        Tool(name="EnvViewer", func=show_environment, description="Displays the current system environment."),
        Tool(name="PackageManager", func=install_pkg, description="Manages package installations.")
    ]

def log_cmd_execution(cmd):
    logger.info(f"Executing command: {cmd}")
    output, error = run_command(cmd)
    if error:
        logger.error(f"Execution failed: {error}")
        return False
    logger.success(f"Execution successful: {output}")
    return True

def retry_cmd(agent, cmd, attempt, max_attempts):
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
    agent = create_agent(tools=setup_tools())
    retry_cmd(agent, cmd, 0, max_attempts)

def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_function
def start_process(cmd, max_attempts):
    logger.info("Initiating process...")
    execute_with_retry(cmd, max_attempts)

def log_cmd(cmd):
    try:
        with open("command_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except Exception as e:
        logger.error(f"Logging command failed: {e}")

def countdown(seconds):
    if seconds > 0:
        logger.info(f"Time left: {seconds} seconds")
        time.sleep(1)
        countdown(seconds - 1)
    else:
        logger.success("Countdown complete!")

def execute_with_settings(cmd, max_attempts=5, timer_duration=0):
    log_cmd(cmd)
    if timer_duration > 0:
        countdown(timer_duration)
    start_process(cmd, max_attempts)

def backup_logs():
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