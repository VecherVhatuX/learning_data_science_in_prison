import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path
from functools import reduce

PYTHON_CMD = "python3"

def fetch_git_changes(repo_path, commit_hash):
    os.chdir(repo_path)
    return subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)

def identify_changed_functions(diff_content):
    return {
        func.split('(')[0].strip('+') for line in diff_content.split('\n') 
        if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])
    }

def find_test_functions(script_path):
    return [
        {"file": str(script_path), "name": node.name, "calls": [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]}
        for node in parse(script_path.read_text(), filename=str(script_path)).body if isinstance(node, FunctionDef) and "test" in node.name
    ]

def collect_all_tests(test_directory):
    return reduce(lambda acc, file: acc + find_test_functions(file), Path(test_directory).rglob("*.py"), [])

def filter_impacted_tests(test_cases, changed_funcs):
    return [
        {"file": test["file"], "name": test["name"], "called": call} 
        for test in test_cases for call in test["calls"] if call in changed_funcs
    ]

def execute_test(test_file, test_function):
    result = subprocess.run([PYTHON_CMD, "-m", "pytest", f"{test_file}::{test_function}"], capture_output=True, text=True)
    return result.returncode == 0

def create_test_report(test_data, results):
    return {
        "total": len(test_data),
        "passed": sum(results),
        "failed": len(test_data) - sum(results),
        "info": [{"name": test["name"], "file": test["file"], "result": "passed" if result else "failed"} for test, result in zip(test_data, results)]
    }

def write_report_to_file(report, output_file):
    with open(output_file, "w") as file:
        json.dump(report, file, indent=2)

def generate_coverage(test_directory):
    subprocess.run([PYTHON_CMD, "-m", "coverage", "run", "--source", test_directory, "-m", "pytest", test_directory])
    subprocess.run([PYTHON_CMD, "-m", "coverage", "html", "-d", "coverage_report"])

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit', required=True, help='Commit hash to compare')
@click.option('--project', required=True, help='Path to the project')
@click.option('--run', is_flag=True, help='Execute the identified tests')
@click.option('--report', is_flag=True, help='Create a test summary')
@click.option('--coverage', is_flag=True, help='Generate coverage report')
@click.option('--output', default="test_summary.json", help='Output file for the summary')
def main(repo, commit, project, run, report, coverage, output):
    changes = fetch_git_changes(repo, commit)
    changed_funcs = identify_changed_functions(changes)
    if not changed_funcs:
        click.echo("No changes in functions detected.")
        return
    
    tests = collect_all_tests(project)
    impacted_tests = filter_impacted_tests(tests, changed_funcs)
    click.echo(json.dumps(impacted_tests, indent=2))
    
    if run:
        outcomes = [execute_test(test['file'], test['name']) for test in impacted_tests]
        for test, result in zip(impacted_tests, outcomes):
            click.echo(f"Executing test: {test['name']} in {test['file']}")
            click.echo(f"Test {test['name']} {'passed' if result else 'failed'}.")
        
        if report:
            summary = create_test_report(impacted_tests, outcomes)
            click.echo(json.dumps(summary, indent=2))
            write_report_to_file(summary, output)
    
    if coverage:
        generate_coverage(project)

if __name__ == "__main__":
    main()