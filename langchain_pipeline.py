import os
import time
import random
from subprocess import run, CalledProcessError
import click
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import reduce

shuffle_data = lambda data: random.sample(data, len(data))

partition_data = lambda data: (
    [entry for entry in data if entry['label'] == 1],
    [entry for entry in data if entry['label'] == 0]
)

update_epoch = lambda data, epoch: (partition_data(shuffle_data(data)), epoch + 1)

show_environment = lambda info: f"Current environment: {dict(os.environ)}\n{info}"

add_package = lambda package: (
    run(["pip", "install", package], text=True, capture_output=True, check=True).stdout
    if not run(["pip", "install", package], text=True, capture_output=True, check=True).returncode
    else run(["pip", "install", package], text=True, capture_output=True, check=True).stderr
)

run_shell_command = lambda command: (
    (run(command, shell=True, text=True, capture_output=True).stdout, run(command, shell=True, text=True, capture_output=True).stderr)
    if command else ("", "Invalid or empty command.")
)

setup_tools = lambda: [
    Tool(name="EnvChecker", func=show_environment, description="Shows current environment settings."),
    Tool(name="PackageManager", func=add_package, description="Installs necessary packages.")
]

execute_with_logging = lambda command: (
    logger.info(f"Running command: {command}") or
    (lambda output, error: (
        logger.error(f"Command failed: {error}") if error else
        logger.success(f"Command succeeded: {output}")
    ))(*run_shell_command(command)) and not error

retry_command = lambda agent, command, attempt, max_attempts: (
    logger.error("Max retries reached. Stopping.") if attempt >= max_attempts else
    logger.info(f"Attempt: {attempt + 1}/{max_attempts}") or
    (execute_with_logging(command) and logger.success("Command executed!")) or
    (agent.run("Check environment variables and dependencies.") and
     agent.run("Try to fix environment issues by installing missing packages.") and
     time.sleep(5) and
     retry_command(agent, command, attempt + 1, max_attempts)
)

run_with_agent = lambda command, max_attempts: (
    lambda tools, agent: retry_command(agent, command, 0, max_attempts)
)(setup_tools(), create_agent(tools=setup_tools()))

time_function = lambda func: lambda *args, **kwargs: (
    lambda start, result: (
        logger.info(f"Time taken: {time.time() - start:.2f} seconds") or result
    )(time.time(), func(*args, **kwargs))
)

begin_process = time_function(lambda command, max_attempts: (
    logger.info("Starting process...") or
    run_with_agent(command, max_attempts)
))

record_command = lambda command: (
    (lambda log: log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n"))(open("command_log.txt", "a"))
    if not IOError else logger.error(f"Logging failed: {IOError}")
)

countdown = lambda seconds: (
    logger.info(f"Countdown: {seconds} seconds left") and
    time.sleep(1) and
    countdown(seconds - 1)
) if seconds > 0 else logger.success("Countdown complete!")

execute = lambda command, max_attempts=5, countdown_time=0: (
    record_command(command) and
    (countdown(countdown_time) if countdown_time > 0 else None) and
    begin_process(command, max_attempts)
)

if __name__ == "__main__":
    click.command()(execute)()