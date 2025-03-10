import os
import subprocess
import time
import click
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseModel
from langchain.agents import tool


# Action to check environment variables using LangChain
class CheckEnvVariables(BaseModel):
    def run(self, inputs: str) -> str:
        """Check system environment variables and conditions."""
        env_info = os.environ  # Gather environment variables
        return f"Current environment variables: {env_info}\n{inputs}"


# Action to fix the environment if needed
class FixEnvVariables(BaseModel):
    def run(self, inputs: str) -> str:
        """Attempt to fix the environment based on inputs."""
        try:
            subprocess.run(f"pip install {inputs}", shell=True, check=True)
            return f"Successfully installed {inputs}."
        except subprocess.CalledProcessError as e:
            return f"Error installing {inputs}: {str(e)}"


# Function to run a shell command
def execute_command(command):
    """Executes a shell command and returns the output."""
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        return result.stdout, result.stderr
    except Exception as e:
        return None, str(e)


# Function to initialize LangChain tools and actions
def initialize_tools():
    """Initialize LangChain tools for checking and fixing environment variables."""
    check_env = CheckEnvVariables()
    fix_env = FixEnvVariables()
    
    check_tool = Tool(
        name="CheckEnvVariables",
        func=check_env.run,
        description="Check the current environment variables to see if the system is ready for the target command."
    )
    
    fix_tool = Tool(
        name="FixEnvVariables",
        func=fix_env.run,
        description="Fix the environment by installing missing packages or adjusting variables."
    )
    
    return [check_tool, fix_tool]


# Function to run the target command
def run_target_command(target_command):
    """Run the target command and check if it executes successfully."""
    print(f"Running target command: {target_command}")
    output, error = execute_command(target_command)
    if error:
        print(f"Error executing command: {error}")
        return False
    print(f"Command output: {output}")
    return True


# LangChain's agent loop to check and fix environment until success
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

    attempts = 0
    while attempts < max_attempts:
        print(f"Attempt {attempts + 1}/{max_attempts}")
        
        # Run the target command first
        if run_target_command(target_command):
            print("Target command executed successfully!")
            break
        
        # Otherwise, the agent decides to check or fix the environment
        agent.run("Check the environment variables and dependencies.")
        
        # Attempt to fix the environment if needed
        agent.run("Try to fix the environment by installing missing dependencies or adjusting variables.")
        
        attempts += 1
        if attempts < max_attempts:
            print("Retrying after checking and fixing the environment...\n")
            time.sleep(5)  # Wait before the next attempt
        else:
            print("Maximum attempts reached. Stopping.")
            break


# Click CLI interface for running the script
@click.command()
@click.argument("target_command")
@click.option("--max_attempts", default=5, help="Maximum number of retry attempts.")
def main(target_command, max_attempts):
    """Main function to run the process with Click CLI."""
    print("Starting process...")
    process_with_agent(target_command, max_attempts)

if __name__ == "__main__":
    main()
