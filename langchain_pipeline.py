import os
import time
import random
from subprocess import run, CalledProcessError
import typer
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger

def randomize_samples(samples: List[Dict]) -> List[Dict]:
    return random.sample(samples, len(samples))

def split_samples(samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    return (
        [item for item in samples if item['label'] == 1],
        [item for item in samples if item['label'] == 0]
    )

def increment_epoch(samples: List[Dict], epoch: int) -> Tuple[Tuple[List[Dict], List[Dict]], int]:
    return split_samples(randomize_samples(samples)), epoch + 1

def display_environment(data: str) -> str:
    return f"Current configurations: {dict(os.environ)}\n{data}"

def install_package(data: str) -> str:
    try:
        return run(
            ["pip", "install", data],
            text=True,
            capture_output=True,
            check=True
        ).stdout
    except CalledProcessError as error:
        return error.stderr

def execute_command(cmd: str) -> Tuple[str, str]:
    if not cmd:
        return "", "Command is empty or invalid."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools() -> List[Tool]:
    return [
        Tool(name="EnvironmentInspector", func=display_environment, description="Shows current configuration details."),
        Tool(name="DependencyInstaller", func=install_package, description="Installs necessary packages.")
    ]

def run_command_with_feedback(cmd: str) -> bool:
    logger.info(f"Executing command: {cmd}")
    output, error = execute_command(cmd)
    if error:
        logger.error(f"Execution failed: {error}")
        return False
    logger.success(f"Command executed successfully: {output}")
    return True

def attempt_command(agent, cmd: str, attempt: int, max_retries: int) -> bool:
    if attempt >= max_retries:
        logger.error("Reached maximum retries. Aborting.")
        return False
    logger.info(f"Attempt number: {attempt + 1}/{max_retries}")
    if run_command_with_feedback(cmd):
        logger.success("Command executed successfully!")
        return True
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing required packages.")
    time.sleep(5)
    return attempt_command(agent, cmd, attempt + 1, max_retries)

def execute_with_agent(cmd: str, max_retries: int):
    tools = initialize_tools()
    agent = create_agent(tools=tools)
    attempt_command(agent, cmd, 0, max_retries)

def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Duration: {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

@time_tracker
def start_process(cmd: str, max_retries: int):
    logger.info("Process is starting...")
    execute_with_agent(cmd, max_retries)

def log_command(cmd: str):
    try:
        with open("command_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except IOError as e:
        logger.error(f"Failed to log command: {e}")

def timer(seconds: int):
    def timer_helper(sec):
        if sec > 0:
            logger.info(f"Timer: {sec} seconds remaining")
            time.sleep(1)
            timer_helper(sec - 1)
        else:
            logger.success("Timer finished!")

    timer_helper(seconds)

def main(cmd: str, max_retries: int = 5, countdown_time: int = 0):
    log_command(cmd)
    if countdown_time > 0:
        timer(countdown_time)
    start_process(cmd, max_retries)

if __name__ == "__main__":
    typer.run(main)