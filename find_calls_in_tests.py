import os
import subprocess
import json
import click
from libcst import parse_module, FunctionDef, Call, Name

PYTHON_EXEC = "python3"

def fetch_modified_functions(repo_dir, commit_id):
    os.chdir(repo_dir)
    git_diff = subprocess.check_output(["git", "diff", commit_id, "--", "*.py"], text=True)
    return {line.split()[3].split('(')[0].strip('+') for line in git_diff.split('\n') 
            if line.startswith('@@') and len(line.split()) > 3 and '(' in line.split()[3]}

def locate_test_functions(project_root):
    test_functions = []
    for root_dir, _, filenames in os.walk(project_root):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(root_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    module_content = parse_module(file.read())
                for node in module_content.body:
                    if isinstance(node, FunctionDef) and "test" in node.name.value:
                        function_calls = [n.func.value for n in module_content.body if isinstance(n, Call) and isinstance(n.func, Name)]
                        test_functions.append({
                            "file_path": file_path,
                            "test_function": node.name.value,
                            "function_calls": function_calls
                        })
    return test_functions

def identify_affected_tests(project_root, modified_functions):
    test_functions = locate_test_functions(project_root)
    return [{
        "file_path": test["file_path"],
        "test_function": test["test_function"],
        "called_function": call
    } for test in test_functions for call in test["function_calls"] if call in modified_functions]

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit', required=True, help='Commit hash to analyze')
@click.option('--project', required=True, help='Path to the project root')
def main(repo, commit, project):
    modified_functions = fetch_modified_functions(repo, commit)
    if not modified_functions:
        click.echo("No functions were modified.")
        return
    click.echo(json.dumps(identify_affected_tests(project, modified_functions), indent=2))

if __name__ == "__main__":
    main()