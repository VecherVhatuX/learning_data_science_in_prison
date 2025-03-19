import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path
from functools import reduce

PYTHON_CMD = "python3"

def fetch_git_changes(repo_path, commit_id):
    os.chdir(repo_path)
    return subprocess.check_output(["git", "diff", commit_id, "--", "*.py"], text=True)

def identify_changed_functions(diff_text):
    return {func.split('(')[0].strip('+') for line in diff_text.split('\n') if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])}

def inspect_python_file(file_path):
    return [{"file": str(file_path), "name": node.name, "calls": [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]}
            for node in parse(file_path.read_text(), filename=str(file_path)).body if isinstance(node, FunctionDef) and "test" in node.name]

def collect_all_tests(directory):
    return reduce(lambda acc, file: acc + inspect_python_file(file), Path(directory).rglob("*.py"), [])

def detect_impacted_tests(test_list, modified_functions):
    return [{"file": test["file"], "name": test["name"], "called": call} for test in test_list for call in test["calls"] if call in modified_functions]

def execute_test(test_path, test_name):
    result = subprocess.run([PYTHON_CMD, "-m", "pytest", f"{test_path}::{test_name}"], capture_output=True, text=True)
    return result.returncode == 0

def compile_test_results(test_list, results):
    return {
        "total": len(test_list),
        "passed": sum(results),
        "failed": len(test_list) - sum(results),
        "info": [{"name": test["name"], "file": test["file"], "result": "passed" if result else "failed"} for test, result in zip(test_list, results)]
    }

def save_results(data, file_name):
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
    changes = fetch_git_changes(repo, commit)
    functions = identify_changed_functions(changes)
    if not functions:
        click.echo("No changes detected in methods.")
        return
    all_tests = collect_all_tests(project)
    affected = detect_impacted_tests(all_tests, functions)
    click.echo(json.dumps(affected, indent=2))
    
    if run:
        results = [execute_test(test['file'], test['name']) for test in affected]
        for test, success in zip(affected, results):
            click.echo(f"Executing test: {test['name']} in {test['file']}")
            click.echo(f"Test {test['name']} {'passed' if success else 'failed'}.")
        
        if report:
            report_data = compile_test_results(affected, results)
            click.echo(json.dumps(report_data, indent=2))
            save_results(report_data, output)

if __name__ == "__main__":
    main()