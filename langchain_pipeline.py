import os
import time
import random
from subprocess import run, CalledProcessError
import click
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import reduce

def shuffle_data(data):
    return random.sample(data, len(data))

def partition_data(data):
    return (
        [entry for entry in data if entry['label'] == 1],
        [entry for entry in data if entry['label'] == 0]
    )

def update_epoch(data, epoch):
    return (partition_data(shuffle_data(data)), epoch + 1)

def show_environment(info):
    return f"Current environment: {dict(os.environ)}\n{info}"

def add_package(package):
    try:
        result = run(["pip", "install", package], text=True, capture_output=True, check=True)
        return result.stdout
    except CalledProcessError as e:
        return e.stderr

def run_shell_command(command):
    if not command:
        return "", "Invalid or empty command."
    result = run(command, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def setup_tools():
    return [
        Tool(name="EnvChecker", func=show_environment, description="Shows current environment settings."),
        Tool(name="PackageManager", func=add_package, description="Installs necessary packages.")
    ]

def execute_with_logging(command):
    logger.info(f"Running command: {command}")
    output, error = run_shell_command(command)
    if error:
        logger.error(f"Command failed: {error}")
    else:
        logger.success(f"Command succeeded: {output}")
    return not error

def retry_command(agent, command, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Max retries reached. Stopping.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if execute_with_logging(command):
        logger.success("Command executed!")
        return
    agent.run("Check environment variables and dependencies.")
    agent.run("Try to fix environment issues by installing missing packages.")
    time.sleep(5)
    retry_command(agent, command, attempt + 1, max_attempts)

def run_with_agent(command, max_attempts):
    tools = setup_tools()
    agent = create_agent(tools=tools)
    retry_command(agent, command, 0, max_attempts)

def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_function
def begin_process(command, max_attempts):
    logger.info("Starting process...")
    run_with_agent(command, max_attempts)

def record_command(command):
    try:
        with open("command_log.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
    except IOError as e:
        logger.error(f"Logging failed: {e}")

def countdown(seconds):
    if seconds > 0:
        logger.info(f"Countdown: {seconds} seconds left")
        time.sleep(1)
        countdown(seconds - 1)
    else:
        logger.success("Countdown complete!")

def execute(command, max_attempts=5, countdown_time=0):
    record_command(command)
    if countdown_time > 0:
        countdown(countdown_time)
    begin_process(command, max_attempts)

if __name__ == "__main__":
    click.command()(execute)()