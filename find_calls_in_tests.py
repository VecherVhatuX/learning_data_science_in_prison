import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path
from functools import reduce

PYTHON_CMD = "python3"

def get_git_diff(repo_path, commit_id):
    os.chdir(repo_path)
    return subprocess.check_output(["git", "diff", commit_id, "--", "*.py"], text=True)

def extract_methods_from_diff(diff_output):
    return {func.split('(')[0].strip('+') for line in diff_output.split('\n') if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])}

def find_test_functions(py_file):
    return [{"file": str(py_file), "name": node.name, "calls": [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]}
            for node in parse(py_file.read_text(), filename=str(py_file)).body if isinstance(node, FunctionDef) and "test" in node.name]

def gather_tests(project_path):
    return reduce(lambda acc, py_file: acc + find_test_functions(py_file), Path(project_path).rglob("*.py"), [])

def filter_affected_tests(tests, modified_methods):
    return [{"file": test["file"], "name": test["name"], "called": call} for test in tests for call in test["calls"] if call in modified_methods]

def run_pytest(test_file, test_name):
    result = subprocess.run([PYTHON_CMD, "-m", "pytest", f"{test_file}::{test_name}"], capture_output=True, text=True)
    return result.returncode == 0

def generate_report_data(affected_tests, outcomes):
    return {
        "total": len(affected_tests),
        "passed": sum(outcomes),
        "failed": len(affected_tests) - sum(outcomes),
        "info": [{"name": test["name"], "file": test["file"], "result": "passed" if result else "failed"} for test, result in zip(affected_tests, outcomes)]
    }

def write_report(report_data, output_file):
    with open(output_file, "w") as file:
        json.dump(report_data, file, indent=2)

@click.command()
@click.option('--repo', required=True, help='Repository path')
@click.option('--commit', required=True, help='Commit hash')
@click.option('--project', required=True, help='Project directory')
@click.option('--run', is_flag=True, help='Run affected tests')
@click.option('--report', is_flag=True, help='Generate test report')
@click.option('--output', default="test_report.json", help='Output file for the test report')
def main(repo, commit, project, run, report, output):
    diff_output = get_git_diff(repo, commit)
    modified_methods = extract_methods_from_diff(diff_output)
    if not modified_methods:
        click.echo("No method changes found.")
        return
    tests = gather_tests(project)
    affected_tests = filter_affected_tests(tests, modified_methods)
    click.echo(json.dumps(affected_tests, indent=2))
    
    if run:
        outcomes = [run_pytest(test['file'], test['name']) for test in affected_tests]
        for test, success in zip(affected_tests, outcomes):
            click.echo(f"Running test: {test['name']} in {test['file']}")
            click.echo(f"Test {test['name']} {'succeeded' if success else 'failed'}.")
        
        if report:
            report_data = generate_report_data(affected_tests, outcomes)
            click.echo(json.dumps(report_data, indent=2))
            write_report(report_data, output)

if __name__ == "__main__":
    main()