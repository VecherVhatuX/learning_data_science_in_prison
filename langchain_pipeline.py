import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

randomize_and_split = lambda items: (
    lambda shuffled: (
        [item for item in shuffled if item['label'] == 1],
        [item for item in shuffled if item['label'] == 0]
    )
)(random.sample(items, len(items)))

process_and_update = lambda items, step: (randomize_and_split(items), step + 1)

display_environment = lambda info: f"System environment details: {dict(os.environ)}\n{info}"

install_package = lambda package: run(
    ["pip", "install", package], text=True, capture_output=True, check=True
).stdout

execute_command = lambda cmd: (
    ("", "No valid command provided.") if not cmd else
    (lambda result: (result.stdout, result.stderr))(
        run(cmd, shell=True, text=True, capture_output=True)
    )
)

initialize_tools = lambda: [
    Tool(name="EnvViewer", func=display_environment, description="Shows the current system environment."),
    Tool(name="PackageManager", func=install_package, description="Handles package installations.")
]

log_execution = lambda cmd: (
    lambda output, error: (
        logger.error(f"Execution failed: {error}") or False if error else
        logger.success(f"Execution successful: {output}") or True
    )
)(*execute_command(cmd)) if logger.info(f"Executing command: {cmd}") else None

retry_execution = lambda agent, cmd, attempt, max_attempts: (
    logger.error("Max retry attempts reached. Stopping.") if attempt >= max_attempts else
    logger.info(f"Retry attempt: {attempt + 1}/{max_attempts}") and
    (logger.success("Command completed successfully!") if log_execution(cmd) else
     agent.run("Check environment variables and dependencies.") and
     agent.run("Fix environment issues by installing missing packages.") and
     time.sleep(5) and
    retry_execution(agent, cmd, attempt + 1, max_attempts)
)

execute_with_retries = lambda cmd, max_attempts: retry_execution(create_agent(tools=initialize_tools()), cmd, 0, max_attempts

time_execution = lambda func: lambda *args, **kwargs: (
    lambda start, result: (
        logger.info(f"Time taken: {time.time() - start:.2f} seconds") or result
    )(time.time(), func(*args, **kwargs))
)

begin_process = time_execution(lambda cmd, max_attempts: (
    logger.info("Initiating process...") and execute_with_retries(cmd, max_attempts)
)

log_command = lambda cmd: (
    lambda: open("command_log.txt", "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")
)() if not logger.error(f"Logging command failed: {e}") else None

start_countdown = lambda seconds: (
    logger.info(f"Time left: {seconds} seconds") and
    time.sleep(1) and start_countdown(seconds - 1) if seconds > 0 else logger.success("Countdown complete!")

execute_with_config = lambda cmd, max_attempts=5, timer_duration=0: (
    log_command(cmd) and
    (start_countdown(timer_duration) if timer_duration > 0 else None) and
    begin_process(cmd, max_attempts)
)

if __name__ == "__main__":
    typer.run(execute_with_config)