import os
import time
import random
from rich.console import Console
from subprocess import run, CalledProcessError
import typer
from tool_library import Tool, create_agent

console = Console()

def shuffle_data(samples):
    random.shuffle(samples)
    return samples

def categorize_samples(samples):
    pos_samples = [item for item in samples if item['label'] == 1]
    neg_samples = [item for item in samples if item['label'] == 0]
    return pos_samples, neg_samples

def update_epoch(samples, epoch):
    epoch += 1
    samples = shuffle_data(samples)
    return categorize_samples(samples), epoch

def show_env_info(data: str) -> str:
    return f"Current environment settings: {dict(os.environ)}\n{data}"

def package_installer(data: str) -> str:
    try:
        result = run(
            ["pip", "install", data],
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout
    except CalledProcessError as error:
        return error.stderr

def run_shell_command(cmd: str) -> tuple:
    result = run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout, result.stderr

def setup_tools() -> list:
    return [
        Tool(name="EnvironmentChecker", func=show_env_info, description="Displays current environment settings."),
        Tool(name="PackageInstaller", func=package_installer, description="Installs required packages.")
    ]

def execute_with_feedback(cmd: str) -> bool:
    console.print(f"Running command: {cmd}")
    output, error = run_shell_command(cmd)
    if error:
        console.print(f"Error during execution: {error}", style="bold red")
        return False
    console.print(f"Successfully executed: {output}", style="bold green")
    return True

def retry_command(agent, cmd: str, attempt: int, max_retries: int) -> bool:
    if attempt < max_retries:
        console.print(f"Attempt: {attempt + 1}/{max_retries}")
        if execute_with_feedback(cmd):
            console.print("Command was successful!", style="bold green")
            return True
        agent.run("Check environment variables and dependencies.")
        agent.run("Try to fix environment issues by installing necessary packages.")
        time.sleep(5)
        return retry_command(agent, cmd, attempt + 1, max_retries)
    console.print("Maximum attempts reached. Stopping execution.", style="bold red")
    return False

def run_with_agent(cmd: str, max_retries: int):
    tools = setup_tools()
    agent = create_agent(tools=tools)
    retry_command(agent, cmd, 0, max_retries)

def timing_decorator(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        console.print(f"Execution time: {time.time() - start:.2f} seconds", style="bold yellow")
        return result
    return inner

@timing_decorator
def initiate_process(cmd: str, max_retries: int):
    console.print("Starting process...", style="bold blue")
    run_with_agent(cmd, max_retries)

def record_command(cmd: str):
    with open("command_log.txt", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd}\n")

def countdown(seconds: int):
    for second in range(seconds, 0, -1):
        console.print(f"Countdown: {second} seconds left", style="bold magenta")
        time.sleep(1)
    console.print("Countdown completed!", style="bold green")

@app.command()
def main(cmd: str, max_retries: int = 5, countdown_time: int = 0):
    record_command(cmd)
    if countdown_time > 0:
        countdown(countdown_time)
    initiate_process(cmd, max_retries)

if __name__ == "__main__":
    typer.run(main)