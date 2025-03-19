import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path
from functools import reduce

PYTHON_CMD = "python3"

def fetch_diff(repo_path, commit_id):
    os.chdir(repo_path)
    return subprocess.check_output(["git", "diff", commit_id, "--", "*.py"], text=True)

def parse_diff(diff_output):
    return {func.split('(')[0].strip('+') for line in diff_output.split('\n') if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])}

def scan_tests(py_file):
    return [{"file": str(py_file), "name": node.name, "calls": [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]}
            for node in parse(py_file.read_text(), filename=str(py_file)).body if isinstance(node, FunctionDef) and "test" in node.name]

def collect_tests(project_path):
    return reduce(lambda acc, py_file: acc + scan_tests(py_file), Path(project_path).rglob("*.py"), [])

def filter_tests(tests, modified_methods):
    return [{"file": test["file"], "name": test["name"], "called": call} for test in tests for call in test["calls"] if call in modified_methods]

def execute_test(test_file, test_name):
    result = subprocess.run([PYTHON_CMD, "-m", "pytest", f"{test_file}::{test_name}"], capture_output=True, text=True)
    return result.returncode == 0

def create_report(affected_tests, outcomes):
    return {
        "total": len(affected_tests),
        "passed": sum(outcomes),
        "failed": len(affected_tests) - sum(outcomes),
        "info": [{"name": test["name"], "file": test["file"], "result": "passed" if result else "failed"} for test, result in zip(affected_tests, outcomes)]
    }

def save_report(report_data, output_file):
    with open(output_file, "w") as file:
        json.dump(report_data, file, indent=2)

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit', required=True, help='Commit identifier')
@click.option('--project', required=True, help='Path to the project directory')
@click.option('--run', is_flag=True, help='Execute affected tests')
@click.option('--report', is_flag=True, help='Create a test report')
@click.option('--output', default="test_report.json", help='Destination file for the test report')
def main(repo, commit, project, run, report, output):
    diff_output = fetch_diff(repo, commit)
    modified_methods = parse_diff(diff_output)
    if not modified_methods:
        click.echo("No changes detected in methods.")
        return
    tests = collect_tests(project)
    affected_tests = filter_tests(tests, modified_methods)
    click.echo(json.dumps(affected_tests, indent=2))
    
    if run:
        outcomes = [execute_test(test['file'], test['name']) for test in affected_tests]
        for test, success in zip(affected_tests, outcomes):
            click.echo(f"Executing test: {test['name']} in {test['file']}")
            click.echo(f"Test {test['name']} {'passed' if success else 'failed'}.")
        
        if report:
            report_data = create_report(affected_tests, outcomes)
            click.echo(json.dumps(report_data, indent=2))
            save_report(report_data, output)

if __name__ == "__main__":
    main()