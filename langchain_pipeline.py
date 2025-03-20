import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

def randomize_and_separate(items):
    random.shuffle(items)
    return [i for i in items if i['label'] == 1], [i for i in items if i['label'] == 0]

def handle_items(items, counter):
    return randomize_and_separate(items), counter + 1

def fetch_system_info(info):
    return f"System environment: {dict(os.environ)}\n{info}"

def add_package(pkg):
    return run(["pip", "install", pkg], text=True, capture_output=True, check=True).stdout

def execute_command(cmd):
    if not cmd:
        return "", "No command provided."
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def initialize_tools():
    return [
        Tool(name="EnvInspector", func=fetch_system_info, description="Shows system environment details."),
        Tool(name="PkgInstaller", func=add_package, description="Handles package installations.")
    ]

def log_command_execution(cmd):
    logger.info(f"Running command: {cmd}")
    stdout, stderr = execute_command(cmd)
    if stderr:
        logger.error(f"Command error: {stderr}")
        return False
    logger.success(f"Command output: {stdout}")
    return True

def retry_command(agent, cmd, attempt, max_attempts):
    if attempt >= max_attempts:
        logger.error("Max retries exceeded. Stopping.")
        return False
    logger.info(f"Retry {attempt + 1} of {max_attempts}")
    if log_command_execution(cmd):
        logger.success("Command successful!")
        return True
    agent.run("Check environment variables and dependencies.")
    agent.run("Install missing packages if needed.")
    time.sleep(5)
    return retry_command(agent, cmd, attempt + 1, max_attempts)

def execute_with_retries(cmd, max_attempts):
    return retry_command(create_agent(tools=initialize_tools()), cmd, 0, max_attempts)

def measure_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@measure_execution_time  # TODO: Typo in decorator name, should be `measure_execution_time`
def initiate_process(cmd, max_attempts):
    logger.info("Process initiated...")
    return execute_with_retries(cmd, max_attempts)

def record_command(cmd):
    try:
        with open("command_log.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
        logger.success("Command logged!")
    except Exception as e:
        logger.error(f"Logging failed: {e}")

def show_countdown(duration):
    if duration > 0:
        logger.info(f"Remaining time: {duration} seconds")
        time.sleep(1)
        show_countdown(duration - 1)
    else:
        logger.success("Countdown finished!")

def execute_with_config(cmd, max_attempts=5, countdown_duration=0):
    record_command(cmd)
    if countdown_duration > 0:
        show_countdown(countdown_duration)
    return initiate_process(cmd, max_attempts)

def create_log_backup():
    try:
        with open("command_log.txt", "r") as f:
            logs = f.read()
        with open(f"command_log_backup_{time.strftime('%Y%m%d%H%M%S')}.txt", "w") as f:
            f.write(logs)
        logger.success("Backup completed!")
    except Exception as e:
        logger.error(f"Backup error: {e}")

def send_notification(message):
    logger.info(f"Sending notification: {message}")
    return run(["notify-send", "Script Notification", message], text=True, capture_output=True).stdout

def check_disk_usage():
    usage = run(["df", "-h"], text=True, capture_output=True).stdout
    logger.info(f"Disk usage:\n{usage}")

def check_network_connection():
    result = run(["ping", "-c", "1", "google.com"], text=True, capture_output=True)
    if result.returncode == 0:
        logger.success("Network connection is active.")
    else:
        logger.error("Network connection is down.")

if __name__ == "__main__":
    typer.run(execute_with_config)
    create_log_backup()
    send_notification("Script execution completed!")
    check_disk_usage()
    check_network_connection()