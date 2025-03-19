import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_and_separate(items):
    shuffled = random.sample(items, len(items))
    return [item for item in shuffled if item['label'] == 1], [item for item in shuffled if item['label'] == 0]

def handle_items(items, count):
    return randomize_and_separate(items), count + 1

def fetch_system_info(info):
    return f"System environment details: {dict(os.environ)}\n{info}"

def add_package(pkg):
    result = run(["pip", "install", pkg], text=True, capture_output=True, check=True)
    return result.stdout

def execute_command(cmd):
    if not cmd:
        return "", "No valid command provided."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools():
    return [
        Tool(name="EnvViewer", func=fetch_system_info, description="Displays system environment details."),
        Tool(name="PackageManager", func=add_package, description="Manages package installations.")
    ]

def log_and_run(cmd):
    logger.info(f"Executing command: {cmd}")
    stdout, stderr = execute_command(cmd)
    if stderr:
        logger.error(f"Command failed: {stderr}")
        return False
    logger.success(f"Command succeeded: {stdout}")
    return True

def retry_command(agent, cmd, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Max retries reached. Aborting.")
        return
    logger.info(f"Attempt {attempt + 1} of {max_attempts}")
    if log_and_run(cmd):
        logger.success("Command executed successfully!")
    else:
        agent.run("Verify environment variables and dependencies.")
        agent.run("Resolve issues by installing required packages.")
        time.sleep(5)
        retry_command(agent, cmd, attempt + 1, max_attempts)

def execute_with_retries(cmd, max_attempts):
    agent = create_agent(tools=initialize_tools())
    retry_command(agent, cmd, 0, max_attempts)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@measure_time
def begin_process(cmd, max_attempts):
    logger.info("Starting process...")
    execute_with_retries(cmd, max_attempts)

def record_command(cmd):
    try:
        with open("command_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except Exception as e:
        logger.error(f"Failed to log command: {e}")

def show_countdown(duration):
    if duration > 0:
        logger.info(f"Time left: {duration} seconds")
        time.sleep(1)
        show_countdown(duration - 1)
    else:
        logger.success("Countdown complete!")

def run_with_config(cmd, max_attempts=5, countdown_duration=0):
    record_command(cmd)
    if countdown_duration > 0:
        show_countdown(countdown_duration)
    begin_process(cmd, max_attempts)

def save_logs():
    try:
        with open("command_log.txt", "r") as log_file:
            logs = log_file.read()
        with open(f"command_log_backup_{time.strftime('%Y%m%d%H%M%S')}.txt", "w") as backup_file:
            backup_file.write(logs)
        logger.success("Backup successful!")
    except Exception as e:
        logger.error(f"Backup failed: {e}")

if __name__ == "__main__":
    typer.run(run_with_config)
    save_logs()