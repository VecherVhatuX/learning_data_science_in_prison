import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_and_filter(items):
    randomized = random.sample(items, len(items))
    return [item for item in randomized if item['label'] == 1], [item for item in randomized if item['label'] == 0]

def increment_and_process(items, iteration):
    return randomize_and_filter(items), iteration + 1

def display_env(info):
    return f"Environment details: {dict(os.environ)}\n{info}"

def install_package(package):
    return run(["pip", "install", package], text=True, capture_output=True, check=True).stdout

def execute_command_shell(cmd):
    if not cmd:
        return "", "Command is empty or invalid."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools():
    return [
        Tool(name="EnvViewer", func=display_env, description="Displays the current environment settings."),
        Tool(name="PackageManager", func=install_package, description="Installs required packages.")
    ]

def log_command_execution(cmd):
    logger.info(f"Running command: {cmd}")
    output, error = execute_command_shell(cmd)
    if error:
        logger.error(f"Command failed: {error}")
        return False
    logger.success(f"Command succeeded: {output}")
    return True

def retry_command_execution(agent, cmd, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Maximum retries reached. Aborting.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if log_command_execution(cmd):
        logger.success("Command executed successfully!")
        return
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing missing packages.")
    time.sleep(5)
    retry_command_execution(agent, cmd, attempt + 1, max_attempts)

def execute_with_retry_logic(cmd, max_attempts):
    agent = create_agent(tools=initialize_tools())
    return retry_command_execution(agent, cmd, 0, max_attempts)

def time_execution(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_execution
def begin_execution(cmd, max_attempts):
    logger.info("Starting process...")
    return execute_with_retry_logic(cmd, max_attempts)

def log_command_history(cmd):
    try:
        with open("command_history.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except IOError as e:
        logger.error(f"Failed to log command: {e}")

def start_countdown(seconds):
    if seconds > 0:
        logger.info(f"Timer: {seconds} seconds remaining")
        time.sleep(1)
        start_countdown(seconds - 1)
    else:
        logger.success("Timer finished!")

def execute_with_options(cmd, max_attempts=5, timer_duration=0):
    log_command_history(cmd)
    if timer_duration > 0:
        start_countdown(timer_duration)
    begin_execution(cmd, max_attempts)

if __name__ == "__main__":
    typer.run(execute_with_options)