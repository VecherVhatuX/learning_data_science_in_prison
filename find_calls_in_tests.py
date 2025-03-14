import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name

PYTHON_INTERPRETER = "python3"

def get_changed_functions(repo_path, commit_hash):
    os.chdir(repo_path)
    diff_output = subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)
    return {
        func.split('(')[0].strip('+') 
        for line in diff_output.split('\n') 
        if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])
    }

def find_test_methods(project_base):
    tests = []
    for dirpath, _, files in os.walk(project_base):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(dirpath, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    ast_tree = parse(f.read(), filename=full_path)
                for node in ast_tree.body:
                    if isinstance(node, FunctionDef) and "test" in node.name:
                        calls = [
                            n.func.id 
                            for n in ast_tree.body 
                            if isinstance(n, Call) and isinstance(n.func, Name)
                        ]
                        tests.append({
                            "path": full_path,
                            "test_name": node.name,
                            "calls": calls
                        })
    return tests

def determine_impacted_tests(project_base, changed_funcs):
    tests = find_test_methods(project_base)
    return [
        {
            "path": test["path"],
            "test_name": test["test_name"],
            "called_func": call
        } 
        for test in tests 
        for call in test["calls"] 
        if call in changed_funcs
    ]

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit', required=True, help='Commit hash to analyze')
@click.option('--project', required=True, help='Base directory of the project')
def execute(repo, commit, project):
    changed_funcs = get_changed_functions(repo, commit)
    if not changed_funcs:
        click.echo("No function changes found.")
        return
    click.echo(json.dumps(determine_impacted_tests(project, changed_funcs), indent=2))

if __name__ == "__main__":
    execute()