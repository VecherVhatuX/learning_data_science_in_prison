import os
import time
import random
from subprocess import run, CalledProcessError
import typer
from typing import List, Dict, Tuple
from tool_library import Tool, create_agent
from loguru import logger
from functools import reduce

# Shuffle the order of elements in the given list
randomize_samples = lambda samples: random.sample(samples, len(samples))

# Separate items into two lists based on their label value
split_samples = lambda samples: (
    [item for item in samples if item['label'] == 1],
    [item for item in samples if item['label'] == 0]
)

# Randomize samples, split them, and increment the epoch counter
increment_epoch = lambda samples, epoch: (split_samples(randomize_samples(samples)), epoch + 1)

# Display the current environment settings along with additional data
display_environment = lambda data: f"Current configurations: {dict(os.environ)}\n{data}"

# Install a package using pip and return the output or error
install_package = lambda data: (
    run(["pip", "install", data], text=True, capture_output=True, check=True).stdout
    if not run(["pip", "install", data], text=True, capture_output=True).returncode
    else run(["pip", "install", data], text=True, capture_output=True).stderr
)

# Execute a shell command and return its output and error
execute_command = lambda cmd: (
    (run(cmd, shell=True, text=True, capture_output=True).stdout, 
    run(cmd, shell=True, text=True, capture_output=True).stderr
) if cmd else ("", "Command is empty or invalid.")

# Initialize a list of tools with specific functionalities
initialize_tools = lambda: [
    Tool(name="EnvironmentInspector", func=display_environment, description="Displays current environment settings."),
    Tool(name="DependencyInstaller", func=install_package, description="Installs required dependencies.")
]

# Run a command and log its success or failure
run_command_with_feedback = lambda cmd: (
    logger.info(f"Executing command: {cmd}") or
    (lambda output, error: (
        logger.error(f"Execution failed: {error}") or False
        if error else logger.success(f"Command executed successfully: {output}") or True
    ))(*execute_command(cmd))
)

# Attempt to execute a command with retries and handle failures
attempt_command = lambda agent, cmd, attempt, max_retries: (
    (logger.error("Reached maximum retries. Aborting.") or False)
    if attempt >= max_retries else
    (logger.info(f"Attempt number: {attempt + 1}/{max_retries}") or
    (run_command_with_feedback(cmd) and (logger.success("Command executed successfully!") or True))
    or (agent.run("Verify environment variables and dependencies.") or
    agent.run("Attempt to resolve environment issues by installing required packages.") or
    time.sleep(5) or attempt_command(agent, cmd, attempt + 1, max_retries)))
)

# Execute a command using an agent with a specified number of retries
execute_with_agent = lambda cmd, max_retries: (
    lambda tools, agent: attempt_command(agent, cmd, 0, max_retries)
)(initialize_tools(), create_agent(tools=initialize_tools()))

# Measure the execution time of a function
time_tracker = lambda func: lambda *args, **kwargs: (
    (lambda start_time: (
        (lambda result: logger.info(f"Duration: {time.time() - start_time:.2f} seconds") or result
        )(func(*args, **kwargs)))
    )(time.time())
)

# Start a process and track its execution time
start_process = time_tracker(lambda cmd, max_retries: (
    logger.info("Process is starting...") or execute_with_agent(cmd, max_retries)
))

# Log a command to a file for future reference
log_command = lambda cmd: (
    (lambda log_file: log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n"))
    (open("command_log.txt", "a")) if not IOError else logger.error(f"Failed to log command: {e}")
)

# Countdown timer that logs remaining time
timer = lambda seconds: (
    (lambda sec: (
        logger.info(f"Timer: {sec} seconds remaining") or time.sleep(1) or timer_helper(sec - 1)
        if sec > 0 else logger.success("Timer finished!")
    )(seconds)
)

# Main function to log, countdown, and start the process
main = lambda cmd, max_retries=5, countdown_time=0: (
    log_command(cmd) or (timer(countdown_time) if countdown_time > 0 else None) or start_process(cmd, max_retries)
)

if __name__ == "__main__":
    typer.run(main)