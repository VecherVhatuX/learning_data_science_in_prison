import os
import time
import random
from subprocess import run, CalledProcessError
import click
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

randomize_items = lambda items: random.sample(items, len(items))

separate_by_label = lambda items: (
    [item for item in items if item['label'] == 1],
    [item for item in items if item['label'] == 0]
)

process_epoch = lambda items, epoch: (separate_by_label(randomize_items(items)), epoch + 1)

display_env = lambda info: f"Environment details: {dict(os.environ)}\n{info}"

install_package = lambda pkg: (
    run(["pip", "install", pkg], text=True, capture_output=True, check=True).stdout
    if not run(["pip", "install", pkg], text=True, capture_output=True, check=True).returncode
    else run(["pip", "install", pkg], text=True, capture_output=True, check=True).stderr
)

execute_shell_cmd = lambda cmd: (
    (run(cmd, shell=True, text=True, capture_output=True).stdout, run(cmd, shell=True, text=True, capture_output=True).stderr)
    if cmd else ("", "Command is empty or invalid.")
)

initialize_tools = lambda: [
    Tool(name="EnvViewer", func=display_env, description="Displays the current environment settings."),
    Tool(name="PackageManager", func=install_package, description="Installs required packages.")
]

log_cmd_exec = lambda cmd: (
    logger.info(f"Running command: {cmd}") or
    (lambda output, error: (
        logger.error(f"Command failed: {error}") if error else logger.success(f"Command succeeded: {output}"),
        not error
    ))(*execute_shell_cmd(cmd))
)

retry_cmd_exec = lambda agent, cmd, attempt, max_attempts: (
    logger.error("Maximum retries reached. Aborting.") if attempt >= max_attempts else
    (logger.info(f"Attempt: {attempt + 1}/{max_attempts}") or
    (log_cmd_exec(cmd) and logger.success("Command executed successfully!") or
    (agent.run("Verify environment variables and dependencies.") or
    agent.run("Attempt to resolve environment issues by installing missing packages.") or
    time.sleep(5) or
    retry_cmd_exec(agent, cmd, attempt + 1, max_attempts)))
)

execute_with_retries = lambda cmd, max_attempts: (
    lambda tools, agent: retry_cmd_exec(agent, cmd, 0, max_attempts)
)(initialize_tools(), create_agent(tools=initialize_tools()))

time_execution = lambda func: (
    lambda *args, **kwargs: (
        lambda start, result: (
            logger.info(f"Execution time: {time.time() - start:.2f} seconds") or result
        )(time.time(), func(*args, **kwargs))
)

start_process = time_execution(lambda cmd, max_attempts: (
    logger.info("Starting process...") or execute_with_retries(cmd, max_attempts)
)

log_cmd = lambda cmd: (
    (lambda log: log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n"))(open("command_history.txt", "a"))
    if not IOError else logger.error(f"Failed to log command: {IOError}")
)

countdown = lambda seconds: (
    (logger.info(f"Timer: {seconds} seconds remaining") or time.sleep(1) or countdown(seconds - 1))
    if seconds > 0 else logger.success("Timer finished!")
)

run_cmd = lambda cmd, max_attempts=5, timer_duration=0: (
    log_cmd(cmd) or (countdown(timer_duration) if timer_duration > 0 else None) or start_process(cmd, max_attempts)
)

if __name__ == "__main__":
    click.command()(run_cmd)()