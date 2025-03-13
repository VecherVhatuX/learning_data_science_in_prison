import os
import subprocess
import json
import click
from parso import parse
from parso.python.tree import Function

PY_LANGUAGE = "python"

def fetch_modified_functions(repo_path, commit_hash):
    os.chdir(repo_path)
    diff = subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)
    modified_funcs = set()
    for line in diff.split('\n'):
        if line.startswith('@@'):
            segments = line.split()
            if len(segments) > 3 and '(' in segments[3]:
                func = segments[3].split('(')[0].strip('+')
                modified_funcs.add(func)
    return modified_funcs

def locate_test_methods(project_dir):
    test_methods = []
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    code = f.read()
                module = parse(code)
                for node in module.iter_funcdefs():
                    if "test" in node.name.value:
                        calls = [call.value for child in node.children if isinstance(child, Function) for call in child.iter_call_names()]
                        test_methods.append({
                            "file": full_path,
                            "test_name": node.name.value,
                            "method_calls": calls
                        })
    return test_methods

def identify_affected_tests(project_dir, modified_funcs):
    test_methods = locate_test_methods(project_dir)
    affected_tests = [{
        "file": test["file"],
        "test_name": test["test_name"],
        "called_function": call
    } for test in test_methods for call in test["method_calls"] if call in modified_funcs]
    return affected_tests

@click.command()
@click.option('--repo', required=True, help='Directory of the repository')
@click.option('--commit_id', required=True, help='ID of the commit to inspect')
@click.option('--project_path', required=True, help='Directory of the project')
def cli(repo, commit_id, project_path):
    modified_funcs = fetch_modified_functions(repo, commit_id)
    if not modified_funcs:
        click.echo("No functions were modified.")
        return
    affected_tests = identify_affected_tests(project_path, modified_funcs)
    click.echo(json.dumps(affected_tests, indent=2))

if __name__ == "__main__":
    cli()