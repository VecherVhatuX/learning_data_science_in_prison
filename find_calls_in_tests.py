import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path
from functools import reduce

# Define the Python executable to use
PYTHON_EXEC = "python3"

# Lambda function to fetch git changes in Python files for a given commit hash
fetch_git_changes = lambda repo_dir, commit_hash: (
    os.chdir(repo_dir),
    subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)
)[1]

# Lambda function to identify changed functions from git diff content
identify_changed_functions = lambda diff_content: {
    func.split('(')[0].strip('+') for line in diff_content.split('\n') 
    if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])
}

# Lambda function to inspect a Python script and extract test functions and their calls
inspect_python_script = lambda script_path: [
    {"file": str(script_path), "name": node.name, "calls": [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]}
    for node in parse(script_path.read_text(), filename=str(script_path)).body if isinstance(node, FunctionDef) and "test" in node.name
]

# Lambda function to collect all test functions from a directory
collect_tests = lambda test_dir: reduce(lambda acc, file: acc + inspect_python_script(file), Path(test_dir).rglob("*.py"), [])

# Lambda function to determine which tests are impacted by the changed functions
determine_impacted_tests = lambda tests, changed_funcs: [
    {"file": test["file"], "name": test["name"], "called": call} 
    for test in tests for call in test["calls"] if call in changed_funcs
]

# Lambda function to execute a specific test and return whether it passed
execute_test = lambda test_file, test_func: (
    lambda result: result.returncode == 0
)(subprocess.run([PYTHON_EXEC, "-m", "pytest", f"{test_file}::{test_func}"], capture_output=True, text=True))

# Lambda function to create a summary of test results
create_test_summary = lambda test_data, outcomes: {
    "total": len(test_data),
    "passed": sum(outcomes),
    "failed": len(test_data) - sum(outcomes),
    "info": [{"name": test["name"], "file": test["file"], "result": "passed" if result else "failed"} for test, result in zip(test_data, outcomes)]
}

# Lambda function to store the test summary in a JSON file
store_summary = lambda summary, filename: (
    lambda file: json.dump(summary, file, indent=2)
)(open(filename, "w"))

# Lambda function to generate a test coverage report
generate_test_coverage_report = lambda test_dir: (
    subprocess.run([PYTHON_EXEC, "-m", "coverage", "run", "--source", test_dir, "-m", "pytest", test_dir]),
    subprocess.run([PYTHON_EXEC, "-m", "coverage", "html", "-d", "coverage_report"])
)

# Click command-line interface definition
@click.command()
@click.option('--repo', required=True, help='Repository directory')
@click.option('--commit', required=True, help='Commit hash')
@click.option('--project', required=True, help='Project directory')
@click.option('--run', is_flag=True, help='Run impacted tests')
@click.option('--report', is_flag=True, help='Generate test summary')
@click.option('--coverage', is_flag=True, help='Generate test coverage report')
@click.option('--output', default="test_summary.json", help='Output file for test summary')
def cli(repo, commit, project, run, report, coverage, output):
    # Fetch git changes and identify altered functions
    changes = fetch_git_changes(repo, commit)
    altered_funcs = identify_changed_functions(changes)
    if not altered_funcs:
        click.echo("No function changes detected.")
        return
    
    # Collect all tests and determine impacted tests
    test_collection = collect_tests(project)
    affected_tests = determine_impacted_tests(test_collection, altered_funcs)
    click.echo(json.dumps(affected_tests, indent=2))
    
    # Run impacted tests if the 'run' flag is set
    if run:
        results = [execute_test(test['file'], test['name']) for test in affected_tests]
        for test, outcome in zip(affected_tests, results):
            click.echo(f"Running test: {test['name']} in {test['file']}")
            click.echo(f"Test {test['name']} {'passed' if outcome else 'failed'}.")
        
        # Generate and store test summary if the 'report' flag is set
        if report:
            summary = create_test_summary(affected_tests, results)
            click.echo(json.dumps(summary, indent=2))
            store_summary(summary, output)
    
    # Generate test coverage report if the 'coverage' flag is set
    if coverage:
        generate_test_coverage_report(project)

if __name__ == "__main__":
    cli()