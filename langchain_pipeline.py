import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

shuffle_and_categorize = lambda items: (
    [item for item in random.sample(items, len(items)) if item['label'] == 1],
    [item for item in random.sample(items, len(items)) if item['label'] == 0]
)

process_data = lambda items, iteration: (shuffle_and_categorize(items), iteration + 1

show_environment = lambda info: f"Environment details: {dict(os.environ)}\n{info}"

add_package = lambda package: run(["pip", "install", package], text=True, capture_output=True, check=True).stdout

run_shell_command = lambda cmd: (run(cmd, shell=True, text=True, capture_output=True).stdout, run(cmd, shell=True, text=True, capture_output=True).stderr) if cmd else ("", "Command is empty or invalid.")

setup_tools = lambda: [
    Tool(name="EnvViewer", func=show_environment, description="Displays the current environment settings."),
    Tool(name="PackageManager", func=add_package, description="Installs required packages.")
]

log_shell_command = lambda cmd: (
    logger.info(f"Running command: {cmd}"),
    (lambda output, error: (
        logger.error(f"Command failed: {error}") if error else logger.success(f"Command succeeded: {output}"),
        not error
    ))(*run_shell_command(cmd))
)[1]

retry_shell_command = lambda agent, cmd, attempt, max_attempts: (
    logger.error("Maximum retries reached. Aborting.") if attempt >= max_attempts else (
        logger.info(f"Attempt: {attempt + 1}/{max_attempts}"),
        log_shell_command(cmd) and logger.success("Command executed successfully!") or (
            agent.run("Verify environment variables and dependencies."),
            agent.run("Attempt to resolve environment issues by installing missing packages."),
            time.sleep(5),
            retry_shell_command(agent, cmd, attempt + 1, max_attempts)
        )
    )
)

execute_with_retries = lambda cmd, max_attempts: retry_shell_command(create_agent(tools=setup_tools()), cmd, 0, max_attempts

measure_execution_time = lambda func: lambda *args, **kwargs: (
    (lambda start: (
        (lambda result: (
            logger.info(f"Execution time: {time.time() - start:.2f} seconds"),
            result
        ))(func(*args, **kwargs))
    ))(time.time())
)

start_execution = measure_execution_time(lambda cmd, max_attempts: (
    logger.info("Starting process..."),
    execute_with_retries(cmd, max_attempts)
))

record_command = lambda cmd: (
    (lambda log: (
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n"),
        log.close()
    ))(open("command_history.txt", "a")) if not IOError else logger.error(f"Failed to log command: {e}")

countdown_timer = lambda seconds: (
    logger.info(f"Timer: {seconds} seconds remaining"),
    time.sleep(1),
    countdown_timer(seconds - 1)
) if seconds > 0 else logger.success("Timer finished!")

execute_command = lambda cmd, max_attempts=5, timer_duration=0: (
    record_command(cmd),
    countdown_timer(timer_duration) if timer_duration > 0 else None,
    start_execution(cmd, max_attempts)
)

if __name__ == "__main__":
    typer.run(execute_command)