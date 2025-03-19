import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path

PYTHON_CMD = "python3"

def get_changed_functions(repo_path, commit_id):
    os.chdir(repo_path)
    diff_output = subprocess.check_output(["git", "diff", commit_id, "--", "*.py"], text=True)
    return {func.split('(')[0].strip('+') for line in diff_output.split('\n') if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])}

def find_test_functions(project_path):
    test_functions = []
    for py_file in Path(project_path).rglob("*.py"):
        with open(py_file, "r", encoding="utf-8") as file:
            ast_tree = parse(file.read(), filename=str(py_file))
        for node in ast_tree.body:
            if isinstance(node, FunctionDef) and "test" in node.name:
                called_functions = [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]
                test_functions.append({"file": str(py_file), "name": node.name, "calls": called_functions})
    return test_functions

def get_impacted_tests(project_path, changed_functions):
    test_functions = find_test_functions(project_path)
    return [{"file": test["file"], "name": test["name"], "called": call} for test in test_functions for call in test["calls"] if call in changed_functions]

def run_test(test_file, test_name):
    result = subprocess.run([PYTHON_CMD, "-m", "pytest", f"{test_file}::{test_name}"], capture_output=True, text=True)
    return result.returncode == 0

def generate_test_summary(impacted_tests, test_results):
    summary = {
        "total": len(impacted_tests),
        "passed": sum(test_results),
        "failed": len(impacted_tests) - sum(test_results),
        "info": []
    }
    for test, result in zip(impacted_tests, test_results):
        summary["info"].append({
            "name": test["name"],
            "file": test["file"],
            "result": "passed" if result else "failed"
        })
    return summary

def save_summary_to_file(summary, file_path):
    with open(file_path, "w") as file:
        json.dump(summary, file, indent=2)

@click.command()
@click.option('--repo', required=True, help='Repository path')
@click.option('--commit', required=True, help='Commit hash')
@click.option('--project', required=True, help='Project directory')
@click.option('--run', is_flag=True, help='Run impacted tests')
@click.option('--report', is_flag=True, help='Generate test summary')
@click.option('--output', default="test_summary.json", help='Output file for the test summary')
def analyze_codebase(repo, commit, project, run, report, output):
    changed_functions = get_changed_functions(repo, commit)
    if not changed_functions:
        click.echo("No function changes found.")
        return
    impacted_tests = get_impacted_tests(project, changed_functions)
    click.echo(json.dumps(impacted_tests, indent=2))
    
    if run:
        outcomes = []
        for test in impacted_tests:
            click.echo(f"Running test: {test['name']} in {test['file']}")
            success = run_test(test['file'], test['name'])
            outcomes.append(success)
            if success:
                click.echo(f"Test {test['name']} succeeded.")
            else:
                click.echo(f"Test {test['name']} failed.")
        
        if report:
            summary_data = generate_test_summary(impacted_tests, outcomes)
            click.echo(json.dumps(summary_data, indent=2))
            save_summary_to_file(summary_data, output)

if __name__ == "__main__":
    analyze_codebase()