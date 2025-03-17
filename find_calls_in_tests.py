import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path

PYTHON_CMD = "python3"

def get_changed_functions(repo_dir, commit_hash):
    os.chdir(repo_dir)
    diff_output = subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)
    return {func.split('(')[0].strip('+') for line in diff_output.split('\n') if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])}

def find_test_functions(project_dir):
    tests = []
    for py_file in Path(project_dir).rglob("*.py"):
        with open(py_file, "r", encoding="utf-8") as file:
            ast_tree = parse(file.read(), filename=str(py_file))
        for node in ast_tree.body:
            if isinstance(node, FunctionDef) and "test" in node.name:
                called_functions = [n.func.id for n in ast_tree.body if isinstance(n, Call) and isinstance(n.func, Name)]
                tests.append({"file": str(py_file), "name": node.name, "calls": called_functions})
    return tests

def find_impacted_tests(project_dir, changed_functions):
    test_functions = find_test_functions(project_dir)
    return [{"file": test["file"], "name": test["name"], "called": call} for test in test_functions for call in test["calls"] if call in changed_functions]

def run_test(test_file, test_name):
    result = subprocess.run([PYTHON_CMD, "-m", "pytest", f"{test_file}::{test_name}"], capture_output=True, text=True)
    return result.returncode == 0

def generate_report(impacted_tests, test_results):
    report = {
        "total": len(impacted_tests),
        "passed": sum(test_results),
        "failed": len(impacted_tests) - sum(test_results),
        "info": []
    }
    for test, result in zip(impacted_tests, test_results):
        report["info"].append({
            "name": test["name"],
            "file": test["file"],
            "result": "passed" if result else "failed"
        })
    return report

@click.command()
@click.option('--repo', required=True, help='Repository path')
@click.option('--commit', required=True, help='Commit hash')
@click.option('--project', required=True, help='Project directory')
@click.option('--run', is_flag=True, help='Run affected tests')
@click.option('--report', is_flag=True, help='Generate test report')
def analyze(repo, commit, project, run, report):
    changed_functions = get_changed_functions(repo, commit)
    if not changed_functions:
        click.echo("No function changes found.")
        return
    impacted_tests = find_impacted_tests(project, changed_functions)
    click.echo(json.dumps(impacted_tests, indent=2))
    
    if run:
        results = []
        for test in impacted_tests:
            click.echo(f"Executing test: {test['name']} in {test['file']}")
            success = run_test(test['file'], test['name'])
            results.append(success)
            if success:
                click.echo(f"Test {test['name']} succeeded.")
            else:
                click.echo(f"Test {test['name']} failed.")
        
        if report:
            report_data = generate_report(impacted_tests, results)
            click.echo(json.dumps(report_data, indent=2))

if __name__ == "__main__":
    analyze()

# TODO: The `find_test_functions` function incorrectly identifies called functions.
# It currently collects all function calls in the entire file, not just within the test function.
# This can lead to false positives when determining impacted tests.
# Fix: Modify the function to only collect calls within the test function's body.