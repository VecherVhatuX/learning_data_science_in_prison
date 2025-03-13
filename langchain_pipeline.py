import os
import time
import random
from subprocess import run, CalledProcessError
import typer
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import reduce

randomize_samples = lambda samples: random.sample(samples, len(samples))

split_samples = lambda samples: (
    [item for item in samples if item['label'] == 1],
    [item for item in samples if item['label'] == 0]
)

increment_epoch = lambda samples, epoch: (split_samples(randomize_samples(samples)), epoch + 1)

display_environment = lambda data: f"Current configurations: {dict(os.environ)}\n{data}"

install_package = lambda data: (
    run(["pip", "install", data], text=True, capture_output=True, check=True).stdout
    if not run(["pip", "install", data], text=True, capture_output=True).returncode
    else run(["pip", "install", data], text=True, capture_output=True).stderr
)

execute_command = lambda cmd: (
    (run(cmd, shell=True, text=True, capture_output=True).stdout, 
    run(cmd, shell=True, text=True, capture_output=True).stderr
) if cmd else ("", "Command is empty or invalid."))

initialize_tools = lambda: [
    Tool(name="EnvironmentInspector", func=display_environment, description="Shows current configuration details."),
    Tool(name="DependencyInstaller", func=install_package, description="Installs necessary packages.")
]

run_command_with_feedback = lambda cmd: (
    logger.info(f"Executing command: {cmd}") or
    (lambda output, error: (
        logger.error(f"Execution failed: {error}") or False
        if error else logger.success(f"Command executed successfully: {output}") or True
    ))(*execute_command(cmd))
)

attempt_command = lambda agent, cmd, attempt, max_retries: (
    (logger.error("Reached maximum retries. Aborting.") or False)
    if attempt >= max_retries else
    (logger.info(f"Attempt number: {attempt + 1}/{max_retries}") or
    (run_command_with_feedback(cmd) and (logger.success("Command executed successfully!") or True))
    or (agent.run("Verify environment variables and dependencies.") or
    agent.run("Attempt to resolve environment issues by installing required packages.") or
    time.sleep(5) or attempt_command(agent, cmd, attempt + 1, max_retries)))
)

execute_with_agent = lambda cmd, max_retries: (
    lambda tools, agent: attempt_command(agent, cmd, 0, max_retries)
)(initialize_tools(), create_agent(tools=initialize_tools()))

time_tracker = lambda func: lambda *args, **kwargs: (
    (lambda start_time: (
        (lambda result: logger.info(f"Duration: {time.time() - start_time:.2f} seconds") or result
        )(func(*args, **kwargs)))
    )(time.time())
)

start_process = time_tracker(lambda cmd, max_retries: (
    logger.info("Process is starting...") or execute_with_agent(cmd, max_retries)
))

log_command = lambda cmd: (
    (lambda log_file: log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n"))
    (open("command_log.txt", "a")) if not IOError else logger.error(f"Failed to log command: {e}")
)

timer = lambda seconds: (
    (lambda sec: (
        logger.info(f"Timer: {sec} seconds remaining") or time.sleep(1) or timer_helper(sec - 1)
        if sec > 0 else logger.success("Timer finished!")
    )(seconds)
)

def backup_data(backup_dir: str = "backup"):
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    backup_file = os.path.join(backup_dir, f"backup_{timestamp}.txt")
    with open(backup_file, "w") as f:
        f.write(f"Backup created at {timestamp}\n")
        f.write("Current environment variables:\n")
        for key, value in dict(os.environ).items():
            f.write(f"{key}={value}\n")
    logger.success(f"Backup created successfully at {backup_file}")

main = lambda cmd, max_retries=5, countdown_time=0: (
    log_command(cmd) or (timer(countdown_time) if countdown_time > 0 else None) or start_process(cmd, max_retries)
)

if __name__ == "__main__":
    typer.run(main)