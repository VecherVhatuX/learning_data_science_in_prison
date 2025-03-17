import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def shuffle_and_separate(items):
    shuffled = random.sample(items, len(items))
    return [item for item in shuffled if item['label'] == 1], [item for item in shuffled if item['label'] == 0]

def process_and_increment(items, step):
    return shuffle_and_separate(items), step + 1

def show_environment(info):
    return f"Environment info: {dict(os.environ)}\n{info}"

def add_package(package):
    return run(["pip", "install", package], text=True, capture_output=True, check=True).stdout

def run_shell_command(cmd):
    if not cmd:
        return "", "Command is empty or invalid."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def setup_tools():
    return [
        Tool(name="EnvDisplay", func=show_environment, description="Shows the current environment settings."),
        Tool(name="PackageInstaller", func=add_package, description="Installs necessary packages.")
    ]

def log_command_run(cmd):
    logger.info(f"Executing command: {cmd}")
    output, error = run_shell_command(cmd)
    if error:
        logger.error(f"Command failed: {error}")
        return False
    logger.success(f"Command succeeded: {output}")
    return True

def retry_command(agent, cmd, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Max retries reached. Stopping.")
        return
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
    if log_command_run(cmd):
        logger.success("Command executed successfully!")
        return
    agent.run("Check environment variables and dependencies.")
    agent.run("Try to fix environment issues by installing missing packages.")
    time.sleep(5)
    retry_command(agent, cmd, attempt + 1, max_attempts)

def execute_with_retry(cmd, max_attempts):
    agent = create_agent(tools=setup_tools())
    return retry_command(agent, cmd, 0, max_attempts)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@measure_time
def start_process(cmd, max_attempts):
    logger.info("Starting execution...")
    return execute_with_retry(cmd, max_attempts)

def save_command(cmd):
    try:
        with open("command_log.txt", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except IOError as e:
        logger.error(f"Failed to save command: {e}")

def countdown(seconds):
    if seconds > 0:
        logger.info(f"Countdown: {seconds} seconds left")
        time.sleep(1)
        countdown(seconds - 1)
    else:
        logger.success("Countdown complete!")

def run_with_settings(cmd, max_attempts=5, timer_duration=0):
    save_command(cmd)
    if timer_duration > 0:
        countdown(timer_duration)
    start_process(cmd, max_attempts)

if __name__ == "__main__":
    typer.run(run_with_settings)