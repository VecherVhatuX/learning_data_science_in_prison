import os
import subprocess
import json
import click
from libcst import parse_module, FunctionDef, Call, Name

PYTHON = "python"

def get_changed_functions(repo_path, commit_hash):
    os.chdir(repo_path)
    diff = subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)
    return {line.split()[3].split('(')[0].strip('+') for line in diff.split('\n') 
            if line.startswith('@@') and len(line.split()) > 3 and '(' in line.split()[3]}

def find_test_functions(project_dir):
    tests = []
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    module = parse_module(f.read())
                for node in module.body:
                    if isinstance(node, FunctionDef) and "test" in node.name.value:
                        calls = [n.func.value for n in module.body if isinstance(n, Call) and isinstance(n.func, Name)]
                        tests.append({
                            "file": full_path,
                            "test_name": node.name.value,
                            "calls": calls
                        })
    return tests

def find_impacted_tests(project_dir, changed_funcs):
    tests = find_test_functions(project_dir)
    return [{
        "file": test["file"],
        "test_name": test["test_name"],
        "called_function": call
    } for test in tests for call in test["calls"] if call in changed_funcs]

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit', required=True, help='Commit hash to analyze')
@click.option('--project', required=True, help='Path to the project root')
def cli(repo, commit, project):
    changed_funcs = get_changed_functions(repo, commit)
    if not changed_funcs:
        click.echo("No functions were modified.")
        return
    click.echo(json.dumps(find_impacted_tests(project, changed_funcs), indent=2))

if __name__ == "__main__":
    cli()