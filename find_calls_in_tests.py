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

def run_tests(test_path, test_name):
    result = subprocess.run([PYTHON_EXECUTABLE, "-m", "pytest", f"{test_path}::{test_name}"], capture_output=True, text=True)
    return result.returncode == 0

def generate_test_report(impacted_tests, results):
    report = {
        "total_tests": len(impacted_tests),
        "passed_tests": sum(results),
        "failed_tests": len(impacted_tests) - sum(results),
        "details": []
    }
    for test, result in zip(impacted_tests, results):
        report["details"].append({
            "test_name": test["test_name"],
            "path": test["path"],
            "status": "passed" if result else "failed"
        })
    return report

@click.command()
@click.option('--repo', required=True, help='Path to the repository')
@click.option('--commit', required=True, help='Commit hash to analyze')
@click.option('--project', required=True, help='Base directory of the project')
@click.option('--run-tests', is_flag=True, help='Run impacted tests automatically')
@click.option('--generate-report', is_flag=True, help='Generate a test report')
def execute_analysis(repo, commit, project, run_tests, generate_report):
    changed_funcs = get_changed_functions(repo, commit)
    if not changed_funcs:
        click.echo("No function changes detected.")
        return
    impacted_tests = find_impacted_tests(project, changed_funcs)
    click.echo(json.dumps(impacted_tests, indent=2))
    
    if run_tests:
        results = []
        for test in impacted_tests:
            click.echo(f"Running test: {test['test_name']} in {test['path']}")
            success = run_tests(test['path'], test['test_name'])
            results.append(success)
            if success:
                click.echo(f"Test {test['test_name']} passed.")
            else:
                click.echo(f"Test {test['test_name']} failed.")
        
        if generate_report:
            report = generate_test_report(impacted_tests, results)
            click.echo(json.dumps(report, indent=2))

if __name__ == "__main__":
    execute_analysis()