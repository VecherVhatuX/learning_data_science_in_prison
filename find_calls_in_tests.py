import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path
from functools import reduce

PYTHON_CMD = "python3"

def get_git_diff(repo_path, commit_id):
    """
    Retrieves the git diff output for Python files in the repository at the specified commit.
    
    Args:
        repo_path (str): Path to the repository.
        commit_id (str): Commit identifier to compare against.
    
    Returns:
        str: The git diff output as a string.
    """
    os.chdir(repo_path)
    return subprocess.check_output(["git", "diff", commit_id, "--", "*.py"], text=True)

def extract_modified_functions(diff_text):
    """
    Extracts the names of functions that were modified in the git diff.
    
    Args:
        diff_text (str): The git diff output.
    
    Returns:
        set: A set of function names that were modified.
    """
    return {func.split('(')[0].strip('+') for line in diff_text.split('\n') if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])}

def analyze_python_file(file_path):
    """
    Analyzes a Python file to extract test functions and the functions they call.
    
    Args:
        file_path (Path): Path to the Python file.
    
    Returns:
        list: A list of dictionaries containing test function names and the functions they call.
    """
    return [{"file": str(file_path), "name": node.name, "calls": [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]}
            for node in parse(file_path.read_text(), filename=str(file_path)).body if isinstance(node, FunctionDef) and "test" in node.name]

def gather_tests(directory):
    """
    Gathers all test functions from Python files in the specified directory.
    
    Args:
        directory (str): Path to the directory containing Python files.
    
    Returns:
        list: A list of dictionaries containing test function information.
    """
    return reduce(lambda acc, file: acc + analyze_python_file(file), Path(directory).rglob("*.py"), [])

def find_affected_tests(test_list, modified_functions):
    """
    Finds test functions that are affected by the modified functions.
    
    Args:
        test_list (list): List of test functions.
        modified_functions (set): Set of modified function names.
    
    Returns:
        list: A list of dictionaries containing affected test functions.
    """
    return [{"file": test["file"], "name": test["name"], "called": call} for test in test_list for call in test["calls"] if call in modified_functions]

def run_test(test_path, test_name):
    """
    Executes a specific test function using pytest.
    
    Args:
        test_path (str): Path to the test file.
        test_name (str): Name of the test function.
    
    Returns:
        bool: True if the test passed, False otherwise.
    """
    result = subprocess.run([PYTHON_CMD, "-m", "pytest", f"{test_path}::{test_name}"], capture_output=True, text=True)
    return result.returncode == 0

def generate_report(test_list, results):
    """
    Generates a report summarizing the test execution results.
    
    Args:
        test_list (list): List of test functions.
        results (list): List of boolean results indicating test pass/fail.
    
    Returns:
        dict: A dictionary containing the test report.
    """
    return {
        "total": len(test_list),
        "passed": sum(results),
        "failed": len(test_list) - sum(results),
        "info": [{"name": test["name"], "file": test["file"], "result": "passed" if result else "failed"} for test, result in zip(test_list, results)]
    }

def save_report(data, file_name):
    """
    Saves the test report to a JSON file.
    
    Args:
        data (dict): The test report data.
        file_name (str): The name of the file to save the report.
    """
    with open(file_name, "w") as file:
        json.dump(data, file, indent=2)

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit', required=True, help='Commit identifier')
@click.option('--project', required=True, help='Path to the project directory')
@click.option('--run', is_flag=True, help='Execute affected tests')
@click.option('--report', is_flag=True, help='Create a test report')
@click.option('--output', default="test_report.json", help='Destination file for the test report')
def main(repo, commit, project, run, report, output):
    """
    Main function to analyze git diff, find affected tests, and optionally run tests and generate a report.
    
    Args:
        repo (str): Path to the repository.
        commit (str): Commit identifier.
        project (str): Path to the project directory.
        run (bool): Flag to execute affected tests.
        report (bool): Flag to generate a test report.
        output (str): Destination file for the test report.
    """
    diff = get_git_diff(repo, commit)
    modified_funcs = extract_modified_functions(diff)
    if not modified_funcs:
        click.echo("No changes detected in methods.")
        return
    tests = gather_tests(project)
    impacted_tests = find_affected_tests(tests, modified_funcs)
    click.echo(json.dumps(impacted_tests, indent=2))
    
    if run:
        test_results = [run_test(test['file'], test['name']) for test in impacted_tests]
        for test, success in zip(impacted_tests, test_results):
            click.echo(f"Executing test: {test['name']} in {test['file']}")
            click.echo(f"Test {test['name']} {'passed' if success else 'failed'}.")
        
        if report:
            report_data = generate_report(impacted_tests, test_results)
            click.echo(json.dumps(report_data, indent=2))
            save_report(report_data, output)

if __name__ == "__main__":
    main()