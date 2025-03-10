import os
import subprocess
import time
import click
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseModel


def check_env_variables(inputs: str) -> str:
    """Check system environment variables and conditions."""
    env_info = os.environ
    return f"Current environment variables: {env_info}\n{inputs}"


def fix_env_variables(inputs: str) -> str:
    """Attempt to fix the environment based on inputs."""
    try:
        subprocess.run(f"pip install {inputs}", shell=True, check=True)
        return f"Successfully installed {inputs}."
    except subprocess.CalledProcessError as e:
        return f"Error installing {inputs}: {str(e)}"


def execute_command(command):
    """Executes a shell command and returns the output."""
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        return result.stdout, result.stderr
    except Exception as e:
        return None, str(e)


def initialize_tools():
    """Initialize LangChain tools for checking and fixing environment variables."""
    return [
        Tool(
            name="CheckEnvVariables",
            func=check_env_variables,
            description="Check the current environment variables to see if the system is ready for the target command."
        ),
        Tool(
            name="FixEnvVariables",
            func=fix_env_variables,
            description="Fix the environment by installing missing packages or adjusting variables."
        )
    ]


def run_target_command(target_command):
    """Run the target command and check if it executes successfully."""
    print(f"Running target command: {target_command}")
    output, error = execute_command(target_command)
    if error:
        print(f"Error executing command: {error}")
        return False
    print(f"Command output: {output}")
    return True


def process_with_agent(target_command, max_attempts):
    """Keep retrying until the command succeeds or max attempts are reached."""
    tools = initialize_tools()
    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")

    agent = initialize_agent(
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        verbose=True
    )

    def attempt_command(attempts):
        if attempts < max_attempts:
            print(f"Attempt {attempts + 1}/{max_attempts}")
            if run_target_command(target_command):
                print("Target command executed successfully!")
                return True
            agent.run("Check the environment variables and dependencies.")
            agent.run("Try to fix the environment by installing missing dependencies or adjusting variables.")
            time.sleep(5)
            return attempt_command(attempts + 1)
        else:
            print("Maximum attempts reached. Stopping.")
            return False

    attempt_command(0)


@click.command()
@click.argument("target_command")
@click.option("--max_attempts", default=5, help="Maximum number of retry attempts.")
def main(target_command, max_attempts):
    """Main function to run the process with Click CLI."""
    print("Starting process...")
    process_with_agent(target_command, max_attempts)


if __name__ == "__main__":
    main()