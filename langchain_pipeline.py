import os
import subprocess
import time
import click
from rich.console import Console
from rich import print
import random
from tool_library import Tool, create_agent

console = Console()

class SampleDataset:
    def __init__(self, data):
        self.data = data
        self.current_epoch = 0

    def randomize_samples(self):
        random.shuffle(self.data)

    def divide_samples(self):
        positives = [item for item in self.data if item['label'] == 1]
        negatives = [item for item in self.data if item['label'] == 0]
        return positives, negatives

    def advance_epoch(self):
        self.current_epoch += 1
        self.randomize_samples()
        return self.divide_samples()

def display_environment_info(input_data: str) -> str:
    return f"Current environment settings: {dict(os.environ)}\n{input_data}"

def install_packages(input_data: str) -> str:
    process = subprocess.run(
        f"pip install {input_data}",
        shell=True,
        text=True,
        capture_output=True
    )
    process.check_returncode()
    return process.stdout

def execute_command(command: str) -> tuple:
    process = subprocess.run(command, shell=True, text=True, capture_output=True)
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

@click.command()
@click.argument("command_to_execute")
@click.option("--max_attempts", default=5, help="Specify the number of retries.")
def main(command_to_execute: str, max_attempts: int):
    log_command(command_to_execute)
    start_process(command_to_execute, max_attempts)

if __name__ == "__main__":
    main()