import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name
from pathlib import Path
from functools import reduce

PYTHON_EXEC = "python3"

fetch_git_changes = lambda repo_dir, commit_hash: (
    os.chdir(repo_dir),
    subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)
)[1]

identify_changed_functions = lambda diff_content: {
    func.split('(')[0].strip('+') for line in diff_content.split('\n') 
    if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])
}

inspect_python_script = lambda script_path: [
    {"file": str(script_path), "name": node.name, "calls": [n.func.id for n in node.body if isinstance(n, Call) and isinstance(n.func, Name)]}
    for node in parse(script_path.read_text(), filename=str(script_path)).body if isinstance(node, FunctionDef) and "test" in node.name
]

collect_tests = lambda test_dir: reduce(lambda acc, file: acc + inspect_python_script(file), Path(test_dir).rglob("*.py"), [])

determine_impacted_tests = lambda tests, changed_funcs: [
    {"file": test["file"], "name": test["name"], "called": call} 
    for test in tests for call in test["calls"] if call in changed_funcs
]

execute_test = lambda test_file, test_func: (
    lambda result: result.returncode == 0
)(subprocess.run([PYTHON_EXEC, "-m", "pytest", f"{test_file}::{test_func}"], capture_output=True, text=True))

create_test_summary = lambda test_data, outcomes: {
    "total": len(test_data),
    "passed": sum(outcomes),
    "failed": len(test_data) - sum(outcomes),
    "info": [{"name": test["name"], "file": test["file"], "result": "passed" if result else "failed"} for test, result in zip(test_data, outcomes)]
}

store_summary = lambda summary, filename: (
    lambda file: json.dump(summary, file, indent=2)
)(open(filename, "w"))

generate_test_coverage_report = lambda test_dir: (
    subprocess.run([PYTHON_EXEC, "-m", "coverage", "run", "--source", test_dir, "-m", "pytest", test_dir]),
    subprocess.run([PYTHON_EXEC, "-m", "coverage", "html", "-d", "coverage_report"])
)

@click.command()
@click.option('--repo', required=True, help='Repository directory')
@click.option('--commit', required=True, help='Commit hash')
@click.option('--project', required=True, help='Project directory')
@click.option('--run', is_flag=True, help='Run impacted tests')
@click.option('--report', is_flag=True, help='Generate test summary')
@click.option('--coverage', is_flag=True, help='Generate test coverage report')
@click.option('--output', default="test_summary.json", help='Output file for test summary')
def cli(repo, commit, project, run, report, coverage, output):
    changes = fetch_git_changes(repo, commit)
    altered_funcs = identify_changed_functions(changes)
    if not altered_funcs:
        click.echo("No function changes detected.")
        return
    test_collection = collect_tests(project)
    affected_tests = determine_impacted_tests(test_collection, altered_funcs)
    click.echo(json.dumps(affected_tests, indent=2))
    
    if run:
        results = [execute_test(test['file'], test['name']) for test in affected_tests]
        for test, outcome in zip(affected_tests, results):
            click.echo(f"Running test: {test['name']} in {test['file']}")
            click.echo(f"Test {test['name']} {'passed' if outcome else 'failed'}.")
        
        if report:
            summary = create_test_summary(affected_tests, results)
            click.echo(json.dumps(summary, indent=2))
            store_summary(summary, output)
    
    if coverage:
        generate_test_coverage_report(project)

if __name__ == "__main__":
    cli()