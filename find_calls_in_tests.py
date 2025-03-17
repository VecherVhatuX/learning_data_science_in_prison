import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path

PYTHON_EXECUTABLE = "python3"

def fetch_modified_methods(repo_path, commit_sha):
    os.chdir(repo_path)
    git_diff = subprocess.check_output(["git", "diff", commit_sha, "--", "*.py"], text=True)
    return {func.split('(')[0].strip('+') for line in git_diff.split('\n') if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])}

def locate_test_methods(project_root):
    test_methods = []
    for file_path in Path(project_root).rglob("*.py"):
        with open(file_path, "r", encoding="utf-8") as f:
            ast_tree = parse(f.read(), filename=str(file_path))
        for node in ast_tree.body:
            if isinstance(node, FunctionDef) and "test" in node.name:
                called_methods = [n.func.id for n in ast_tree.body if isinstance(n, Call) and isinstance(n.func, Name)]
                test_methods.append({"path": str(file_path), "test_name": node.name, "calls": called_methods})
    return test_methods

def identify_affected_tests(project_root, modified_methods):
    test_methods = locate_test_methods(project_root)
    return [{"path": test["path"], "test_name": test["test_name"], "called_func": call} for test in test_methods for call in test["calls"] if call in modified_methods]

def execute_test_method(test_path, test_name):
    result = subprocess.run([PYTHON_EXECUTABLE, "-m", "pytest", f"{test_path}::{test_name}"], capture_output=True, text=True)
    return result.returncode == 0

def create_test_summary(affected_tests, results):
    summary = {
        "total_tests": len(affected_tests),
        "passed_tests": sum(results),
        "failed_tests": len(affected_tests) - sum(results),
        "details": []
    }
    for test, result in zip(affected_tests, results):
        summary["details"].append({
            "test_name": test["test_name"],
            "path": test["path"],
            "status": "passed" if result else "failed"
        })
    return summary

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit', required=True, help='Commit hash to analyze')
@click.option('--project', required=True, help='Base directory of the project')
@click.option('--run-tests', is_flag=True, help='Run impacted tests automatically')
@click.option('--generate-report', is_flag=True, help='Generate a test report')
def perform_analysis(repo, commit, project, run_tests, generate_report):
    modified_methods = fetch_modified_methods(repo, commit)
    if not modified_methods:
        click.echo("No function changes detected.")
        return
    affected_tests = identify_affected_tests(project, modified_methods)
    click.echo(json.dumps(affected_tests, indent=2))
    
    if run_tests:
        results = []
        for test in affected_tests:
            click.echo(f"Running test: {test['test_name']} in {test['path']}")
            success = execute_test_method(test['path'], test['test_name'])
            results.append(success)
            if success:
                click.echo(f"Test {test['test_name']} passed.")
            else:
                click.echo(f"Test {test['test_name']} failed.")
        
        if generate_report:
            summary = create_test_summary(affected_tests, results)
            click.echo(json.dumps(summary, indent=2))

if __name__ == "__main__":
    perform_analysis()