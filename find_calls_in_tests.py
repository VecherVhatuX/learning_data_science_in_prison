import os
import subprocess
import json
import click
from ast import parse, FunctionDef, Call, Name

PYTHON_INTERPRETER = "python3"

def extract_diff_functions(repo_path, commit_hash):
    os.chdir(repo_path)
    diff = subprocess.check_output(["git", "diff", commit_hash, "--", "*.py"], text=True)
    return {func.split('(')[0].strip('+') for line in diff.split('\n') if line.startswith('@@') and len(line.split()) > 3 and '(' in (func := line.split()[3])}

def scan_for_tests(base_dir):
    test_cases = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    ast_tree = parse(f.read(), filename=file_path)
                for node in ast_tree.body:
                    if isinstance(node, FunctionDef) and "test" in node.name:
                        called_functions = [n.func.id for n in ast_tree.body if isinstance(n, Call) and isinstance(n.func, Name)]
                        test_cases.append({"path": file_path, "test_name": node.name, "calls": called_functions})
    return test_cases

def identify_affected_tests(base_dir, modified_functions):
    test_cases = scan_for_tests(base_dir)
    return [{"path": test["path"], "test_name": test["test_name"], "called_func": call} for test in test_cases for call in test["calls"] if call in modified_functions]

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit', required=True, help='Commit hash to analyze')
@click.option('--project', required=True, help='Base directory of the project')
def run_analysis(repo, commit, project):
    modified_functions = extract_diff_functions(repo, commit)
    if not modified_functions:
        click.echo("No function changes found.")
        return
    click.echo(json.dumps(identify_affected_tests(project, modified_functions), indent=2))

if __name__ == "__main__":
    run_analysis()