import os
import subprocess
import json
import click
from tree_sitter import Language, Parser

# Path to the Tree-sitter Python grammar
PY_LANGUAGE = Language("build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)

def get_changed_functions(repo_path, commit_id):
    """
    Find functions that were modified in the given commit.
    
    Args:
        repo_path (str): Path to the Git repository.
        commit_id (str): The commit ID to check for changes.
    
    Returns:
        set: A set of function names that were modified.
    """
    os.chdir(repo_path)
    diff_output = subprocess.check_output(["git", "diff", commit_id, "--", "*.py"], text=True)
    
    changed_functions = set()
    for line in diff_output.split('\n'):
        if line.startswith('@@'):
            parts = line.split()
            if len(parts) > 3 and '(' in parts[3]:
                func_name = parts[3].split('(')[0].strip('+')
                changed_functions.add(func_name)
    return changed_functions

def find_test_functions(project_path):
    """
    Find all test functions (functions containing 'test' in their name) and the method calls inside them.
    
    Args:
        project_path (str): Path to the project folder.
    
    Returns:
        list: A list of dictionaries containing file paths, test function names, and method calls.
    """
    test_functions = []
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                tree = parser.parse(content.encode())
                
                # Traverse the AST to find test function definitions and method calls inside them
                cursor = tree.walk()
                stack = [cursor.node]
                while stack:
                    node = stack.pop()
                    if node.type == "function_definition":
                        func_name = content[node.children[1].start_byte:node.children[1].end_byte]
                        if "test" in func_name:
                            method_calls = []
                            # Find all method calls inside this function
                            for child in node.children:
                                if child.type == "block":
                                    call_stack = [child]
                                    while call_stack:
                                        call_node = call_stack.pop()
                                        if call_node.type == "call":
                                            call_name = content[call_node.children[0].start_byte:call_node.children[0].end_byte]
                                            method_calls.append(call_name)
                                        call_stack.extend(call_node.children)
                            test_functions.append({
                                "file": file_path,
                                "test_name": func_name,
                                "method_calls": method_calls
                            })
                    stack.extend(node.children)
    return test_functions

def find_tests_calling_functions(project_path, changed_functions):
    """
    Identify test functions that call any of the changed functions.
    
    Args:
        project_path (str): Path to the project folder.
        changed_functions (set): A set of function names that were modified.
    
    Returns:
        list: A list of dictionaries containing test file paths, test function names, and the changed functions they call.
    """
    test_functions = find_test_functions(project_path)
    test_results = []
    
    for test in test_functions:
        file_path = test["file"]
        test_name = test["test_name"]
        method_calls = test["method_calls"]
        
        for call in method_calls:
            if call in changed_functions:
                test_results.append({
                    "file": file_path,
                    "test_name": test_name,
                    "called_function": call
                })
    return test_results

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit_id', required=True, help='Commit ID to check')
@click.option('--project_path', required=True, help='Path to project folder')
def main(repo, commit_id, project_path):
    """
    Main CLI function.
    
    Fetches changed functions from the given commit, finds test functions in the project,
    and identifies which tests call the changed functions. Outputs the results in JSON format.
    """
    changed_functions = get_changed_functions(repo, commit_id)
    if not changed_functions:
        click.echo("No changed functions found.")
        return
    
    test_results = find_tests_calling_functions(project_path, changed_functions)
    click.echo(json.dumps(test_results, indent=2))

if __name__ == "__main__":
    main()
