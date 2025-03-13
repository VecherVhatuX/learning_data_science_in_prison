import os
import subprocess
import json
import click
from parso import parse
from parso.python.tree import Function

PYTHON_LANG = "python"

def fetch_modified_functions(repo_dir, commit_ref):
    os.chdir(repo_dir)
    diff_output = subprocess.check_output(["git", "diff", commit_ref, "--", "*.py"], text=True)
    return {segment[3].split('(')[0].strip('+') for line in diff_output.split('\n') 
            if line.startswith('@@') and len(segment := line.split()) > 3 and '(' in segment[3]}

def locate_test_methods(project_root):
    test_cases = []
    for dirpath, _, filenames in os.walk(project_root):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    parsed_module = parse(file.read())
                test_cases.extend({
                    "file": file_path,
                    "test_name": func.name.value,
                    "method_calls": [call.value for child in func.children if isinstance(child, Function) for call in child.iter_call_names()]
                } for func in parsed_module.iter_funcdefs() if "test" in func.name.value)
    return test_cases

def identify_affected_tests(project_root, modified_funcs):
    test_methods = locate_test_methods(project_root)
    return [{
        "file": test["file"],
        "test_name": test["test_name"],
        "called_function": call
    } for test in test_methods for call in test["method_calls"] if call in modified_funcs]

@click.command()
@click.option('--repo', required=True, help='Repository directory path')
@click.option('--commit_id', required=True, help='Commit reference to analyze')
@click.option('--project_path', required=True, help='Project root directory path')
def main(repo, commit_id, project_path):
    modified_funcs = fetch_modified_functions(repo, commit_id)
    if not modified_funcs:
        click.echo("No function modifications detected.")
        return
    click.echo(json.dumps(identify_affected_tests(project_path, modified_funcs), indent=2))

if __name__ == "__main__":
    main()