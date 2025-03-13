import os
import time
import random
from rich.console import Console
from subprocess import run, CalledProcessError
import typer
from tool_library import Tool, create_agent

console = Console()

def randomize_samples(samples):
    return random.sample(samples, len(samples))

def split_samples(samples):
    return (
        [item for item in samples if item['label'] == 1],
        [item for item in samples if item['label'] == 0]
    )

def increment_epoch(samples, epoch):
    return split_samples(randomize_samples(samples)), epoch + 1

def display_environment(data: str) -> str:
    return f"Current configurations: {dict(os.environ)}\n{data}"

def install_package(data: str) -> str:
    try:
        return run(
            ["pip", "install", data],
            text=True,
            capture_output=True,
            check=True
        ).stdout
    except CalledProcessError as error:
        return error.stderr

def execute_command(cmd: str) -> tuple:
    return run(cmd, shell=True, text=True, capture_output=True).stdout, run(cmd, shell=True, text=True, capture_output=True).stderr

def initialize_tools() -> list:
    return [
        Tool(name="EnvironmentInspector", func=display_environment, description="Shows current configuration details."),
        Tool(name="DependencyInstaller", func=install_package, description="Installs necessary packages.")
    ]

def run_command_with_feedback(cmd: str) -> bool:
    console.print(f"Executing command: {cmd}")
    output, error = execute_command(cmd)
    if error:
        console.print(f"Execution failed: {error}", style="bold red")
        return False
    console.print(f"Command executed successfully: {output}", style="bold green")
    return True

def attempt_command(agent, cmd: str, attempt: int, max_retries: int) -> bool:
    if attempt >= max_retries:
        console.print("Reached maximum retries. Aborting.", style="bold red")
        return False
    console.print(f"Attempt number: {attempt + 1}/{max_retries}")
    if run_command_with_feedback(cmd):
        console.print("Command executed successfully!", style="bold green")
        return True
    agent.run("Verify environment variables and dependencies.")
    agent.run("Attempt to resolve environment issues by installing required packages.")
    time.sleep(5)
    return attempt_command(agent, cmd, attempt + 1, max_retries)

def execute_with_agent(cmd: str, max_retries: int):
    tools = initialize_tools()
    agent = create_agent(tools=tools)
    attempt_command(agent, cmd, 0, max_retries)

def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        console.print(f"Duration: {time.time() - start_time:.2f} seconds", style="bold yellow")
        return result
    return wrapper

@time_tracker
def start_process(cmd: str, max_retries: int):
    console.print("Process is starting...", style="bold blue")
    execute_with_agent(cmd, max_retries)

def log_command(cmd: str):
    with open("command_log.txt", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")

def timer(seconds: int):
    def timer_helper(sec):
        if sec > 0:
            console.print(f"Timer: {sec} seconds remaining", style="bold magenta")
            time.sleep(1)
            timer_helper(sec - 1)
        else:
            console.print("Timer finished!", style="bold green")

    timer_helper(seconds)

@app.command()
def main(cmd: str, max_retries: int = 5, countdown_time: int = 0):
    log_command(cmd)
    if countdown_time > 0:
        timer(countdown_time)
    start_process(cmd, max_retries)

if __name__ == "__main__":
    typer.run(main)