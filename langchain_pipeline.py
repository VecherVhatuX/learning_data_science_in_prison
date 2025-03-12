import os
import time
import random
from rich.console import Console
from subprocess import run, CalledProcessError
import typer
from tool_library import Tool, create_agent

console = Console()

def randomize_samples(data):
    random.shuffle(data)
    return data

def divide_samples(data):
    positives = [item for item in data if item['label'] == 1]
    negatives = [item for item in data if item['label'] == 0]
    return positives, negatives

def advance_epoch(data, current_epoch):
    current_epoch += 1
    data = randomize_samples(data)
    return divide_samples(data), current_epoch

def display_environment_info(input_data: str) -> str:
    return f"Current environment settings: {dict(os.environ)}\n{input_data}"

def install_packages(input_data: str) -> str:
    try:
        process = run(
            ["pip", "install", input_data],
            text=True,
            capture_output=True,
            check=True
        )
        return process.stdout
    except CalledProcessError as e:
        return e.stderr

def execute_command(command: str) -> tuple:
    process = run(command, shell=True, text=True, capture_output=True)
    return process.stdout, process.stderr

def initialize_tools() -> list:
    return [
        Tool(name="EnvVariableChecker", func=display_environment_info, description="Shows current environment settings."),
        Tool(name="EnvVariableInstaller", func=install_packages, description="Installs necessary packages for environment correction.")
    ]

def run_command_with_feedback(command_to_execute: str) -> bool:
    console.print(f"Executing command: {command_to_execute}")
    output, error_message = execute_command(command_to_execute)
    if error_message:
        console.print(f"Execution encountered an error: {error_message}", style="bold red")
        return False
    console.print(f"Execution completed successfully: {output}", style="bold green")
    return True

def attempt_command(agent, command_to_execute: str, current_attempt: int, max_attempts: int) -> bool:
    if current_attempt < max_attempts:
        console.print(f"Attempt number: {current_attempt + 1}/{max_attempts}")
        if run_command_with_feedback(command_to_execute):
            console.print("Command executed successfully!", style="bold green")
            return True
        agent.run("Assess environment variables and dependencies.")
        agent.run("Attempt to resolve environment challenges by installing necessary packages.")
        time.sleep(5)
        return attempt_command(agent, command_to_execute, current_attempt + 1, max_attempts)
    console.print("Reached maximum attempts. Halting operation.", style="bold red")
    return False

def execute_with_agent(command_to_execute: str, max_attempts: int):
    tools = initialize_tools()
    agent = create_agent(tools=tools)
    attempt_command(agent, command_to_execute, 0, max_attempts)

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        console.print(f"Total execution time: {time.time() - start_time:.2f} seconds", style="bold yellow")
        return result
    return wrapper

@measure_execution_time
def start_process(command_to_execute: str, max_attempts: int):
    console.print("Initializing process...", style="bold blue")
    execute_with_agent(command_to_execute, max_attempts)

def log_command(command: str):
    with open("command_history.log", "a") as history_file:
        history_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")

def countdown_timer(seconds: int):
    for i in range(seconds, 0, -1):
        console.print(f"Countdown: {i} seconds remaining", style="bold magenta")
        time.sleep(1)
    console.print("Countdown has finished!", style="bold green")

@app.command()
def main(command_to_execute: str, max_attempts: int = 5, countdown: int = 0):
    log_command(command_to_execute)
    if countdown > 0:
        countdown_timer(countdown)
    start_process(command_to_execute, max_attempts)

if __name__ == "__main__":
    typer.run(main)