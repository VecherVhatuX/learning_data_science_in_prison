import os
import subprocess
import json
import click
from parso import parse
from parso.python.tree import Function

PY_LANGUAGE = "python"

def get_changed_functions(repo_path, commit_hash):
    os.chdir(repo_path)
    diff = subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)
    return {parts[3].split('(')[0].strip('+') for line in diff.split('\n') 
            if line.startswith('@@') and len(parts := line.split()) > 3 and '(' in parts[3]}

def find_test_functions(project_dir):
    tests = []
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    module = parse(f.read())
                tests.extend({
                    "file": full_path,
                    "test_name": node.name.value,
                    "method_calls": [call.value for child in node.children if isinstance(child, Function) for call in child.iter_call_names()]
                } for node in module.iter_funcdefs() if "test" in node.name.value)
    return tests

def find_impacted_tests(project_dir, changed_funcs):
    tests = find_test_functions(project_dir)
    return [{
        "file": test["file"],
        "test_name": test["test_name"],
        "called_function": call
    } for test in tests for call in test["method_calls"] if call in changed_funcs]

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit_id', required=True, help='Commit hash to analyze')
@click.option('--project_path', required=True, help='Path to the project')
def cli(repo, commit_id, project_path):
    changed_funcs = get_changed_functions(repo, commit_id)
    if not changed_funcs:
        click.echo("No changes detected in functions.")
        return
    click.echo(json.dumps(find_impacted_tests(project_path, changed_funcs), indent=2)

if __name__ == "__main__":
    cli()