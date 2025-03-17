import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_and_sort(items):
    randomized = random.sample(items, len(items))
    return (
        [item for item in randomized if item['label'] == 1],
        [item for item in randomized if item['label'] == 0]
    )

def handle_data(items, iteration):
    return (randomize_and_sort(items), iteration + 1)

def display_environment(info):
    return f"Environment details: {dict(os.environ)}\n{info}"

def install_package(package):
    result = run(["pip", "install", package], text=True, capture_output=True, check=True)
    return result.stdout if not result.returncode else result.stderr

def execute_shell_command(cmd):
    if not cmd:
        return "", "Command is empty or invalid."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools():
    return [
        Tool(name="EnvViewer", func=display_environment, description="Displays the current environment settings."),
        Tool(name="PackageManager", func=install_package, description="Installs required packages.")
    ]

def log_command(cmd):
    logger.info(f"Running command: {cmd}")
    output, error = execute_shell_command(cmd)
    if error:
        logger.error(f"Command failed: {error}")
        return False
    logger.success(f"Command succeeded: {output}")
    return True

def retry_command(agent, cmd, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Maximum retries reached. Aborting.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if log_command(cmd):
        logger.success("Command executed successfully!")
        return
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing missing packages.")
    time.sleep(5)
    retry_command(agent, cmd, attempt + 1, max_attempts)

def execute_with_retry(cmd, max_attempts):
    tools = initialize_tools()
    agent = create_agent(tools=tools)
    retry_command(agent, cmd, 0, max_attempts)

def time_execution(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_execution
def begin_process(cmd, max_attempts):
    logger.info("Starting process...")
    execute_with_retry(cmd, max_attempts)

def log_command_history(cmd):
    try:
        with open("command_history.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except IOError as e:
        logger.error(f"Failed to log command: {e}")

def start_timer(seconds):
    if seconds > 0:
        logger.info(f"Timer: {seconds} seconds remaining")
        time.sleep(1)
        start_timer(seconds - 1)
    else:
        logger.success("Timer finished!")

def run_command(cmd: str, max_attempts: int = 5, timer_duration: int = 0):
    log_command_history(cmd)
    if timer_duration > 0:
        start_timer(timer_duration)
    begin_process(cmd, max_attempts)

if __name__ == "__main__":
    typer.run(run_command)