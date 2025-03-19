import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path

PYTHON_CMD = "python3"

def fetch_modified_methods(repo_path, commit_id):
    os.chdir(repo_path)
    diff_output = subprocess.check_output(["git", "diff", commit_id, "--", "*.py"], text=True)
    return {func.split('(')[0].strip('+') for line in diff_output.split('\n') if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])}

def locate_test_methods(project_path):
    test_methods = []
    for py_file in Path(project_path).rglob("*.py"):
        with open(py_file, "r", encoding="utf-8") as file:
            ast_tree = parse(file.read(), filename=str(py_file))
        for node in ast_tree.body:
            if isinstance(node, FunctionDef) and "test" in node.name:
                called_methods = [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]
                test_methods.append({"file": str(py_file), "name": node.name, "calls": called_methods})
    return test_methods

def identify_affected_tests(project_path, modified_methods):
    test_methods = locate_test_methods(project_path)
    return [{"file": test["file"], "name": test["name"], "called": call} for test in test_methods for call in test["calls"] if call in modified_methods]

def execute_test(test_file, test_name):
    result = subprocess.run([PYTHON_CMD, "-m", "pytest", f"{test_file}::{test_name}"], capture_output=True, text=True)
    return result.returncode == 0

def create_test_report(affected_tests, test_results):
    report = {
        "total": len(affected_tests),
        "passed": sum(test_results),
        "failed": len(affected_tests) - sum(test_results),
        "info": []
    }
    for test, result in zip(affected_tests, test_results):
        report["info"].append({
            "name": test["name"],
            "file": test["file"],
            "result": "passed" if result else "failed"
        })
    return report

def save_report_to_file(report, file_path):
    with open(file_path, "w") as file:
        json.dump(report, file, indent=2)

@click.command()
@click.option('--repo', required=True, help='Repository path')
@click.option('--commit', required=True, help='Commit hash')
@click.option('--project', required=True, help='Project directory')
@click.option('--run', is_flag=True, help='Run affected tests')
@click.option('--report', is_flag=True, help='Generate test report')
@click.option('--output', default="test_report.json", help='Output file for the test report')
def analyze_repository(repo, commit, project, run, report, output):
    modified_methods = fetch_modified_methods(repo, commit)
    if not modified_methods:
        click.echo("No method changes found.")
        return
    affected_tests = identify_affected_tests(project, modified_methods)
    click.echo(json.dumps(affected_tests, indent=2))
    
    if run:
        outcomes = []
        for test in affected_tests:
            click.echo(f"Executing test: {test['name']} in {test['file']}")
            success = execute_test(test['file'], test['name'])
            outcomes.append(success)
            if success:
                click.echo(f"Test {test['name']} succeeded.")
            else:
                click.echo(f"Test {test['name']} failed.")
        
        if report:
            report_data = create_test_report(affected_tests, outcomes)
            click.echo(json.dumps(report_data, indent=2))
            save_report_to_file(report_data, output)

if __name__ == "__main__":
    analyze_repository()