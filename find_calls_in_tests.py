import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name

PYTHON_EXECUTABLE = "python3"

def get_changed_functions(repo_path, commit_sha):
    os.chdir(repo_path)
    git_diff = subprocess.check_output(["git", "diff", commit_sha, "--", "*.py"], text=True)
    return {func.split('(')[0].strip('+') for line in git_diff.split('\n') if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])}

def find_test_functions(project_root):
    tests = []
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    ast_tree = parse(f.read(), filename=full_path)
                for node in ast_tree.body:
                    if isinstance(node, FunctionDef) and "test" in node.name:
                        called_funcs = [n.func.id for n in ast_tree.body if isinstance(n, Call) and isinstance(n.func, Name)]
                        tests.append({"path": full_path, "test_name": node.name, "calls": called_funcs})
    return tests

def find_impacted_tests(project_root, changed_funcs):
    tests = find_test_functions(project_root)
    return [{"path": test["path"], "test_name": test["test_name"], "called_func": call} for test in tests for call in test["calls"] if call in changed_funcs]

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit', required=True, help='Commit hash to analyze')
@click.option('--project', required=True, help='Base directory of the project')
def execute_analysis(repo, commit, project):
    changed_funcs = get_changed_functions(repo, commit)
    if not changed_funcs:
        click.echo("No function changes detected.")
        return
    click.echo(json.dumps(find_impacted_tests(project, changed_funcs), indent=2))

if __name__ == "__main__":
    execute_analysis()