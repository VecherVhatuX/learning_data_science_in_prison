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

def get_system_details(info):
    return f"System environment: {dict(os.environ)}\n{info}"

def install_package(pkg):
    output = run(["pip", "install", pkg], text=True, capture_output=True, check=True)
    return output.stdout

def run_shell_command(cmd):
    if not cmd:
        return "", "No command provided."
    output = run(cmd, shell=True, text=True, capture_output=True)
    return output.stdout, output.stderr

def setup_tools():
    return [
        Tool(name="EnvInspector", func=get_system_details, description="Shows system environment details."),
        Tool(name="PkgInstaller", func=install_package, description="Handles package installations.")
    ]

def log_execution(cmd):
    logger.info(f"Running command: {cmd}")
    stdout, stderr = run_shell_command(cmd)
    if stderr:
        logger.error(f"Command error: {stderr}")
        return False
    logger.success(f"Command output: {stdout}")
    return True

def retry_execution(agent, cmd, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Max retries exceeded. Stopping.")
        return
    logger.info(f"Retry {attempt + 1} of {max_attempts}")
    if log_execution(cmd):
        logger.success("Command successful!")
    else:
        agent.run("Check environment variables and dependencies.")
        agent.run("Install missing packages if needed.")
        time.sleep(5)
        retry_execution(agent, cmd, attempt + 1, max_attempts)

def execute_with_retry(cmd, max_attempts):
    agent = create_agent(tools=setup_tools())
    retry_execution(agent, cmd, 0, max_attempts)

def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@time_execution
def start_process(cmd, max_attempts):
    logger.info("Process initiated...")
    execute_with_retry(cmd, max_attempts)

def log_command(cmd):
    try:
        with open("command_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
    except Exception as e:
        logger.error(f"Logging failed: {e}")

def display_countdown(duration):
    if duration > 0:
        logger.info(f"Remaining time: {duration} seconds")
        time.sleep(1)
        display_countdown(duration - 1)
    else:
        logger.success("Countdown finished!")

def execute_with_settings(cmd, max_attempts=5, countdown_duration=0):
    log_command(cmd)
    if countdown_duration > 0:
        display_countdown(countdown_duration)
    start_process(cmd, max_attempts)

def backup_logs():
    try:
        with open("command_log.txt", "r") as log_file:
            logs = log_file.read()
        with open(f"command_log_backup_{time.strftime('%Y%m%d%H%M%S')}.txt", "w") as backup_file:
            backup_file.write(logs)
        logger.success("Backup completed!")
    except Exception as e:
        logger.error(f"Backup error: {e}")

if __name__ == "__main__":
    typer.run(execute_with_settings)
    backup_logs()