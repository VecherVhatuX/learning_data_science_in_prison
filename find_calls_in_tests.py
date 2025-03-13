import os
import subprocess
import json
import click
from parso import parse
from parso.python.tree import Function

# Define the language for parsing (Python in this case)
PY_LANGUAGE = "python"

# Function to get the list of functions that have changed in a specific commit
def get_changed_functions(repo_path, commit_hash):
    # Change the current working directory to the repository path
    os.chdir(repo_path)
    # Get the git diff output for Python files in the specified commit
    diff = subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)
    # Parse the diff output to extract changed function names
    return {parts[3].split('(')[0].strip('+') for line in diff.split('\n') 
            if line.startswith('@@') and len(parts := line.split()) > 3 and '(' in parts[3]}

# Function to find all test functions in the project directory
def find_test_functions(project_dir):
    tests = []
    # Walk through the project directory to find all Python files
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                # Open and parse the Python file
                with open(full_path, "r", encoding="utf-8") as f:
                    module = parse(f.read())
                # Extract test functions and their method calls
                tests.extend({
                    "file": full_path,
                    "test_name": node.name.value,
                    "method_calls": [call.value for child in node.children if isinstance(child, Function) for call in child.iter_call_names()]
                } for node in module.iter_funcdefs() if "test" in node.name.value)
    return tests

# Function to find tests impacted by the changed functions
def find_impacted_tests(project_dir, changed_funcs):
    # Get all test functions in the project
    tests = find_test_functions(project_dir)
    # Filter tests that call any of the changed functions
    return [{
        "file": test["file"],
        "test_name": test["test_name"],
        "called_function": call
    } for test in tests for call in test["method_calls"] if call in changed_funcs]

# CLI command to analyze impacted tests based on a commit
@click.command()
@click.option('--repo', required=True, help='Location of the repository')
@click.option('--commit_id', required=True, help='Hash of the commit to inspect')
@click.option('--project_path', required=True, help='Directory of the project')
def cli(repo, commit_id, project_path):
    # Get the list of changed functions
    changed_funcs = get_changed_functions(repo, commit_id)
    if not changed_funcs:
        click.echo("No function modifications found.")
        return
    # Find and print the impacted tests in JSON format
    click.echo(json.dumps(find_impacted_tests(project_path, changed_funcs), indent=2)

# Entry point for the script
if __name__ == "__main__":
    cli()