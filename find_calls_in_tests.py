import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path
from functools import reduce

PYTHON_EXEC = "python3"

def get_git_diff(repo_path, commit_id):
    os.chdir(repo_path)
    return subprocess.check_output(["git", "diff", commit_id, "--", "*.py"], text=True)

def extract_modified_functions(diff_output):
    return {
        func.split('(')[0].strip('+') for line in diff_output.split('\n') 
        if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])
    }

def analyze_script(script_path):
    return [
        {"file": str(script_path), "name": node.name, "calls": [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]}
        for node in parse(script_path.read_text(), filename=str(script_path)).body if isinstance(node, FunctionDef) and "test" in node.name
    ]

def gather_tests(test_dir):
    return reduce(lambda acc, file: acc + analyze_script(file), Path(test_dir).rglob("*.py"), [])

def find_affected_tests(tests, modified_funcs):
    return [
        {"file": test["file"], "name": test["name"], "called": call} 
        for test in tests for call in test["calls"] if call in modified_funcs
    ]

def run_test(test_file, test_func):
    result = subprocess.run([PYTHON_EXEC, "-m", "pytest", f"{test_file}::{test_func}"], capture_output=True, text=True)
    return result.returncode == 0

def create_summary(test_data, results):
    return {
        "total": len(test_data),
        "passed": sum(results),
        "failed": len(test_data) - sum(results),
        "info": [{"name": test["name"], "file": test["file"], "result": "passed" if result else "failed"} for test, result in zip(test_data, results)]
    }

def save_summary(summary, output_file):
    with open(output_file, "w") as file:
        json.dump(summary, file, indent=2)

def generate_coverage(test_dir):
    subprocess.run([PYTHON_EXEC, "-m", "coverage", "run", "--source", test_dir, "-m", "pytest", test_dir])
    subprocess.run([PYTHON_EXEC, "-m", "coverage", "html", "-d", "coverage_report"])

@click.command()
@click.option('--repo', required=True, help='Repository directory')
@click.option('--commit', required=True, help='Commit hash')
@click.option('--project', required=True, help='Project directory')
@click.option('--run', is_flag=True, help='Run impacted tests')
@click.option('--report', is_flag=True, help='Generate test summary')
@click.option('--coverage', is_flag=True, help='Generate test coverage report')
@click.option('--output', default="test_summary.json", help='Output file for test summary')
def execute(repo, commit, project, run, report, coverage, output):
    diff = get_git_diff(repo, commit)
    modified = extract_modified_functions(diff)
    if not modified:
        click.echo("No function changes detected.")
        return
    
    tests = gather_tests(project)
    affected = find_affected_tests(tests, modified)
    click.echo(json.dumps(affected, indent=2))
    
    if run:
        results = [run_test(test['file'], test['name']) for test in affected]
        for test, outcome in zip(affected, results):
            click.echo(f"Running test: {test['name']} in {test['file']}")
            click.echo(f"Test {test['name']} {'passed' if outcome else 'failed'}.")
        
        if report:
            summary = create_summary(affected, results)
            click.echo(json.dumps(summary, indent=2))
            save_summary(summary, output)
    
    if coverage:
        generate_coverage(project)

if __name__ == "__main__":
    execute()