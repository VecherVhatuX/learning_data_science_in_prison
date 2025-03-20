import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path
from functools import reduce

PYTHON_EXEC = "python3"

def fetch_git_changes(repo_path, commit_id):
    os.chdir(repo_path)
    return subprocess.check_output(["git", "diff", commit_id, "--", "*.py"], text=True)

def identify_changed_functions(diff_output):
    return {
        func.split('(')[0].strip('+') for line in diff_output.split('\n') 
        if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])
    }

def inspect_script(script_path):
    return [
        {"file": str(script_path), "name": node.name, "calls": [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]}
        for node in parse(script_path.read_text(), filename=str(script_path)).body if isinstance(node, FunctionDef) and "test" in node.name
    ]

def collect_tests(test_directory):
    return reduce(lambda acc, file: acc + inspect_script(file), Path(test_directory).rglob("*.py"), [])

def detect_impacted_tests(test_list, modified_funcs):
    return [
        {"file": test["file"], "name": test["name"], "called": call} 
        for test in test_list for call in test["calls"] if call in modified_funcs
    ]

def execute_test(test_file, test_function):
    result = subprocess.run([PYTHON_EXEC, "-m", "pytest", f"{test_file}::{test_function}"], capture_output=True, text=True)
    return result.returncode == 0

def compile_test_summary(test_data, results):
    return {
        "total": len(test_data),
        "passed": sum(results),
        "failed": len(test_data) - sum(results),
        "info": [{"name": test["name"], "file": test["file"], "result": "passed" if result else "failed"} for test, result in zip(test_data, results)]
    }

def store_summary(summary_data, output_file):
    with open(output_file, "w") as file:
        json.dump(summary_data, file, indent=2)

def generate_test_coverage(test_directory):
    subprocess.run([PYTHON_EXEC, "-m", "coverage", "run", "--source", test_directory, "-m", "pytest", test_directory])
    subprocess.run([PYTHON_EXEC, "-m", "coverage", "html", "-d", "coverage_report"])

@click.command()
@click.option('--repo', required=True, help='Repository directory')
@click.option('--commit', required=True, help='Commit hash')
@click.option('--project', required=True, help='Project directory')
@click.option('--run', is_flag=True, help='Run impacted tests')
@click.option('--report', is_flag=True, help='Generate test summary')
@click.option('--coverage', is_flag=True, help='Generate test coverage report')
@click.option('--output', default="test_summary.json", help='Output file for test summary')
def main(repo, commit, project, run, report, coverage, output):
    diff_output = fetch_git_changes(repo, commit)
    modified_functions = identify_changed_functions(diff_output)
    if not modified_functions:
        click.echo("No function changes detected.")
        return
    
    test_list = collect_tests(project)
    affected_tests = detect_impacted_tests(test_list, modified_functions)
    click.echo(json.dumps(affected_tests, indent=2))
    
    if run:
        test_results = [execute_test(test['file'], test['name']) for test in affected_tests]
        for test, outcome in zip(affected_tests, test_results):
            click.echo(f"Running test: {test['name']} in {test['file']}")
            click.echo(f"Test {test['name']} {'passed' if outcome else 'failed'}.")
        
        if report:
            summary = compile_test_summary(affected_tests, test_results)
            click.echo(json.dumps(summary, indent=2))
            store_summary(summary, output)
    
    if coverage:
        generate_test_coverage(project)

if __name__ == "__main__":
    main()