import os
import subprocess
import json
import click
from tree_sitter import Language, Parser

PY_LANGUAGE = Language("build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)

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
                
                tree = parser.parse(code.encode())
                cursor = tree.walk()
                stack = [cursor.node]
                while stack:
                    node = stack.pop()
                    if node.type == "function_definition":
                        func_name = code[node.children[1].start_byte:node.children[1].end_byte]
                        if "test" in func_name:
                            calls = []
                            for child in node.children:
                                if child.type == "block":
                                    call_stack = [child]
                                    while call_stack:
                                        call_node = call_stack.pop()
                                        if call_node.type == "call":
                                            call_name = code[call_node.children[0].start_byte:call_node.children[0].end_byte]
                                            calls.append(call_name)
                                        call_stack.extend(call_node.children)
                            test_methods.append({
                                "file": full_path,
                                "test_name": func_name,
                                "method_calls": calls
                            })
                    stack.extend(node.children)
    return test_methods

def identify_affected_tests(project_dir, modified_funcs):
    test_methods = locate_test_methods(project_dir)
    affected_tests = []
    
    for test in test_methods:
        file_path = test["file"]
        test_name = test["test_name"]
        calls = test["method_calls"]
        
        for call in calls:
            if call in modified_funcs:
                affected_tests.append({
                    "file": file_path,
                    "test_name": test_name,
                    "called_function": call
                })
    return affected_tests

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit_id', required=True, help='Commit ID to check')
@click.option('--project_path', required=True, help='Path to project folder')
def cli(repo, commit_id, project_path):
    modified_funcs = fetch_modified_functions(repo, commit_id)
    if not modified_funcs:
        click.echo("No modified functions found.")
        return
    
    affected_tests = identify_affected_tests(project_path, modified_funcs)
    click.echo(json.dumps(affected_tests, indent=2))

if __name__ == "__main__":
    cli()