import os
import subprocess
import time
import click
from rich.console import Console
from another_library import Tool, setup_agent
import random

console = Console()

def shuffle_samples(samples):
    """Randomly reorder the provided list of samples."""
    random.shuffle(samples)

def get_positive_and_negative_samples(samples):
    """Separate samples into two lists: positive and negative based on their labels."""
    return (
        [s for s in samples if s['label'] == 1],
        [s for s in samples if s['label'] == 0]
    )

def next_epoch(samples, epochs):
    """Increment the epoch count and shuffle samples for the next round."""
    epochs += 1
    shuffle_samples(samples)
    return epochs, get_positive_and_negative_samples(samples)

def get_env_info(inputs: str) -> str:
    """Retrieve and format the current environment variables along with provided input."""
    return f"Current environment variables: {dict(os.environ)}\n{inputs}"

def install_dependencies(inputs: str) -> str:
    """Run a command to install specified Python packages using pip."""
    result = subprocess.run(
        f"pip install {inputs}",
        shell=True,
        text=True,
        capture_output=True
    )
    result.check_returncode()
    return result.stdout

def run_shell_command(command: str) -> tuple:
    """Execute a shell command and return its output and error streams."""
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def setup_tools() -> list:
    """Initialize a list of tools for managing environment variables."""
    return [
        Tool(name="CheckEnvVariables", func=get_env_info, description="Check current environment variables."),
        Tool(name="FixEnvVariables", func=install_dependencies, description="Fix the environment by installing missing packages.")
    ]

def execute_with_feedback(target_command: str) -> bool:
    """Run a command and provide console feedback based on success or failure."""
    console.print(f"Executing command: {target_command}")
    output, error = run_shell_command(target_command)
    if error:
        console.print(f"Command execution failed: {error}", style="bold red")
        return False
    console.print(f"Command execution result: {output}", style="bold green")
    return True

def retry_command(agent, target_command: str, attempts: int, max_attempts: int) -> bool:
    """Attempt to execute a command multiple times if it fails, providing feedback each time."""
    if attempts < max_attempts:
        console.print(f"Attempt number: {attempts + 1}/{max_attempts}")
        if execute_with_feedback(target_command):
            console.print("Command executed successfully!", style="bold green")
            return True
        agent.run("Review environment variables and dependencies.")
        agent.run("Attempt to resolve environment issues by installing required dependencies.")
        time.sleep(5)
        return retry_command(agent, target_command, attempts + 1, max_attempts)
    console.print("Reached the limit of attempts. Halting operation.", style="bold red")
    return False

def process_command_with_agent(target_command: str, max_attempts: int):
    """Coordinate the command execution process with the help of an agent."""
    tools = setup_tools()
    agent = setup_agent(tools=tools)
    retry_command(agent, target_command, 0, max_attempts)

def log_execution_time(func):
    """Decorator to log the execution duration of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        console.print(f"Total execution time: {time.time() - start_time:.2f} seconds", style="bold yellow")
        return result
    return wrapper

@log_execution_time
def main_process(target_command: str, max_attempts: int):
    """Main function to initiate the command processing."""
    console.print("Process is starting...", style="bold blue")
    process_command_with_agent(target_command, max_attempts)

def log_command_history(command: str):
    """Record the executed command with a timestamp in a log file."""
    with open("command_history.log", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")

@click.command()
@click.argument("target_command")
@click.option("--max_attempts", default=5, help="Maximum number of retry attempts.")
def main(target_command: str, max_attempts: int):
    """Entry point of the program to log the command and initiate processing."""
    log_command_history(target_command)
    main_process(target_command, max_attempts)

if __name__ == "__main__":
    main()