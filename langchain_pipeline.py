import os
import time
import random
from subprocess import run
import typer
from tool_library import Tool, create_agent
from loguru import logger
from functools import wraps

# Lambda function to randomize and separate items based on their 'label' attribute
randomize_and_separate = lambda items: (
    [i for i in random.sample(items, len(items)) if i['label'] == 1],
    [i for i in random.sample(items, len(items)) if i['label'] == 0]
)

# Lambda function to handle items by randomizing, separating, and incrementing a counter
handle_items = lambda items, counter: (randomize_and_separate(items), counter + 1)

# Lambda function to fetch and format system environment information
fetch_system_info = lambda info: f"System environment: {dict(os.environ)}\n{info}"

# Lambda function to install a Python package using pip
add_package = lambda pkg: run(["pip", "install", pkg], text=True, capture_output=True, check=True).stdout

# Lambda function to execute a shell command and return its output and error
execute_command = lambda cmd: (run(cmd, shell=True, text=True, capture_output=True).stdout, run(cmd, shell=True, text=True, capture_output=True).stderr) if cmd else ("", "No command provided.")

# Lambda function to initialize a list of tools with specific functionalities
initialize_tools = lambda: [
    Tool(name="EnvInspector", func=fetch_system_info, description="Shows system environment details."),
    Tool(name="PkgInstaller", func=add_package, description="Handles package installations.")
]

# Lambda function to log the execution of a command, including its output and errors
log_command_execution = lambda cmd: (
    logger.info(f"Running command: {cmd}"),
    (stdout, stderr) := execute_command(cmd),
    logger.error(f"Command error: {stderr}") if stderr else logger.success(f"Command output: {stdout}"),
    not stderr
)[-1]

# Lambda function to retry a command execution up to a maximum number of attempts
retry_command = lambda agent, cmd, attempt, max_attempts: (
    logger.error("Max retries exceeded. Stopping.") if attempt >= max_attempts else (
        logger.info(f"Retry {attempt + 1} of {max_attempts}"),
        log_command_execution(cmd) and logger.success("Command successful!") or (
            agent.run("Check environment variables and dependencies."),
            agent.run("Install missing packages if needed."),
            time.sleep(5),
            retry_command(agent, cmd, attempt + 1, max_attempts)
        )
    )
)

# Lambda function to execute a command with retries
execute_with_retries = lambda cmd, max_attempts: retry_command(create_agent(tools=initialize_tools()), cmd, 0, max_attempts)

# Lambda function to measure the execution time of a given function
measure_execution_time = lambda func: wraps(func)(lambda *args, **kwargs: (
    start := time.time(),
    result := func(*args, **kwargs),
    logger.info(f"Time taken: {time.time() - start:.2f} seconds"),
    result
)[-1]

# Lambda function to initiate a process, measure its execution time, and execute with retries
initiate_process = measure_execution_time(lambda cmd, max_attempts: (
    logger.info("Process initiated..."),
    execute_with_retries(cmd, max_attempts)
))

# Lambda function to record a command execution in a log file
record_command = lambda cmd: (
    (open("command_log.txt", "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n"),
    logger.success("Command logged!")
) if not (e := Exception()) else logger.error(f"Logging failed: {e}")

# Lambda function to display a countdown timer
show_countdown = lambda duration: (
    logger.info(f"Remaining time: {duration} seconds"),
    time.sleep(1),
    show_countdown(duration - 1)
) if duration > 0 else logger.success("Countdown finished!")

# Lambda function to execute a command with configuration options (retries and countdown)
execute_with_config = lambda cmd, max_attempts=5, countdown_duration=0: (
    record_command(cmd),
    show_countdown(countdown_duration) if countdown_duration > 0 else None,
    initiate_process(cmd, max_attempts)
)

# Lambda function to create a backup of the command log file
create_log_backup = lambda: (
    (logs := open("command_log.txt", "r").read()),
    open(f"command_log_backup_{time.strftime('%Y%m%d%H%M%S')}.txt", "w").write(logs),
    logger.success("Backup completed!")
) if not (e := Exception()) else logger.error(f"Backup error: {e}")

# Lambda function to send a system notification
send_notification = lambda message: (
    logger.info(f"Sending notification: {message}"),
    run(["notify-send", "Script Notification", message], text=True, capture_output=True).stdout
)

if __name__ == "__main__":
    typer.run(execute_with_config)
    create_log_backup()
    send_notification("Script execution completed!")