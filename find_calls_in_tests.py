import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name

PYTHON_EXEC = "python3"

def get_changed_functions(repo_path, commit_hash):
    os.chdir(repo_path)
    diff_output = subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)
    return {line.split()[3].split('(')[0].strip('+') for line in diff_output.split('\n') 
            if line.startswith('@@') and len(line.split()) > 3 and '(' in line.split()[3]}

def find_test_functions(project_path):
    tests = []
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    ast_tree = parse(f.read(), filename=full_path)
                for node in ast_tree.body:
                    if isinstance(node, FunctionDef) and "test" in node.name:
                        calls = [n.func.id for n in ast_tree.body if isinstance(n, Call) and isinstance(n.func, Name)]
                        tests.append({
                            "path": full_path,
                            "name": node.name,
                            "calls": calls
                        })
    return tests

def get_impacted_tests(project_path, changed_functions):
    tests = find_test_functions(project_path)
    return [{
        "path": test["path"],
        "test": test["name"],
        "called": call
    } for test in tests for call in test["calls"] if call in changed_functions]

@click.command()
@click.option('--repo', required=True, help='Repository directory')
@click.option('--commit', required=True, help='Commit ID')
@click.option('--project', required=True, help='Project root directory')
def cli(repo, commit, project):
    changed_functions = get_changed_functions(repo, commit)
    if not changed_functions:
        click.echo("No functions changed.")
        return
    click.echo(json.dumps(get_impacted_tests(project, changed_functions), indent=2))

if __name__ == "__main__":
    cli()