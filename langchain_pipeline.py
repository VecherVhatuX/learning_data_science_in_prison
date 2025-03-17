import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_and_split(items):
    shuffled = random.sample(items, len(items))
    return [item for item in shuffled if item['label'] == 1], [item for item in shuffled if item['label'] == 0]

def process_and_update(items, step):
    return randomize_and_split(items), step + 1

def display_environment(info):
    return f"System environment details: {dict(os.environ)}\n{info}"

def install_package(package):
    return run(["pip", "install", package], text=True, capture_output=True, check=True).stdout

def execute_command(cmd):
    if not cmd:
        return "", "No valid command provided."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools():
    return [
        Tool(name="EnvViewer", func=display_environment, description="Shows the current system environment."),
        Tool(name="PackageManager", func=install_package, description="Handles package installations.")
    ]

def log_execution(cmd):
    logger.info(f"Executing command: {cmd}")
    output, error = execute_command(cmd)
    if error:
        logger.error(f"Execution failed: {error}")
        return False
    logger.success(f"Execution successful: {output}")
    return True

def retry_execution(agent, cmd, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Max retry attempts reached. Stopping.")
        return
    logger.info(f"Retry attempt: {attempt + 1}/{max_attempts}")
    if log_execution(cmd):
        logger.success("Command completed successfully!")
        return
    agent.run("Check environment variables and dependencies.")
    agent.run("Fix environment issues by installing missing packages.")
    time.sleep(5)
    retry_execution(agent, cmd, attempt + 1, max_attempts)

def execute_with_retries(cmd, max_attempts):
    agent = create_agent(tools=initialize_tools())
    return retry_execution(agent, cmd, 0, max_attempts)

def time_execution(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_execution
def begin_process(cmd, max_attempts):
    logger.info("Initiating process...")
    return execute_with_retries(cmd, max_attempts)

def log_command(cmd):
    try:
        with open("command_log.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except IOError as e:
        logger.error(f"Logging command failed: {e}")

def start_countdown(seconds):
    if seconds > 0:
        logger.info(f"Time left: {seconds} seconds")
        time.sleep(1)
        start_countdown(seconds - 1)
    else:
        logger.success("Countdown complete!")

def execute_with_config(cmd, max_attempts=5, timer_duration=0):
    log_command(cmd)
    if timer_duration > 0:
        start_countdown(timer_duration)
    begin_process(cmd, max_attempts)

if __name__ == "__main__":
    typer.run(execute_with_config)