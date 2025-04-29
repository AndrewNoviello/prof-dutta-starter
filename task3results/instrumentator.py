import ast
import csv
import re
import importlib
import sys

# Helper: Check if a variable is scalar-like
def generate_logging_code(var_name):
    return [
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[
                    ast.Constant(value=f"log>>>{var_name}:"),
                    ast.Call(
                        func=ast.Name(id='safe_log', ctx=ast.Load()),
                        args=[ast.Name(id=var_name, ctx=ast.Load())],
                        keywords=[]
                    )
                ],
                keywords=[]
            )
        )
    ]

# Helper: Generate code for setting random seeds
def generate_seed_code():
    return [
        ast.Import(names=[ast.alias(name='random', asname=None)]),
        ast.Import(names=[ast.alias(name='numpy', asname='np')]),
        ast.Import(names=[ast.alias(name='torch', asname=None)]),
        ast.parse("random.seed(42)").body[0],
        ast.parse("np.random.seed(42)").body[0],
        ast.parse("torch.manual_seed(42)").body[0],
        ast.parse("torch.cuda.manual_seed_all(42)").body[0]
    ]

# Insert a safe logger function
def insert_safe_log_function(tree):
    safe_log_code = """
def safe_log(x):
    try:
        if hasattr(x, 'shape') or hasattr(x, '__len__'):
            return {'min': x.min(), 'max': x.max()}
        else:
            return x
    except Exception as e:
        return str(x)
"""
    safe_log_ast = ast.parse(safe_log_code)
    tree.body = safe_log_ast.body + tree.body
    return tree

# Helper: Extract variables from an assert expression (basic version)
def extract_assert_vars(expr):
    vars = []
    if isinstance(expr, ast.Compare):
        if isinstance(expr.left, ast.Name):
            vars.append(expr.left.id)
        for comparator in expr.comparators:
            if isinstance(comparator, ast.Name):
                vars.append(comparator.id)
    elif isinstance(expr, ast.Call):
        for arg in expr.args:
            vars.extend(extract_assert_vars(arg))
    elif isinstance(expr, ast.Name):
        vars.append(expr.id)
    return vars

# Main function to instrument a file
def instrumentor(file_path, test_name, assertion_line):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    tree = insert_safe_log_function(tree)  # Insert safe logger

    modified = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            # Insert random seeds at start of test
            seed_code = generate_seed_code()
            node.body = seed_code + node.body

            for idx, stmt in enumerate(node.body):
                if hasattr(stmt, 'lineno') and stmt.lineno == int(assertion_line):
                    if isinstance(stmt, ast.Assert):
                        targets = extract_assert_vars(stmt.test)
                        logging_nodes = []
                        for var in targets:
                            logging_nodes += generate_logging_code(var)
                        node.body = node.body[:idx] + logging_nodes + node.body[idx:]
                        modified = True
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        call = stmt.value
                        targets = []
                        for arg in call.args:
                            targets += extract_assert_vars(arg)
                        logging_nodes = []
                        for var in targets:
                            logging_nodes += generate_logging_code(var)
                        node.body = node.body[:idx] + logging_nodes + node.body[idx:]
                        modified = True

    if modified:
        new_source = ast.unparse(tree)  # <--- use only builtin Python
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_source)
    else:
        print(f"No matching assertion found in {file_path} at line {assertion_line}")


def run_test_n_times(module_name, test_function_name, n_runs=100, output_csv='log_output.csv'):
    logs = []

    for _ in range(n_runs):
        try:
            # Reload the module each time to reset random seeds
            if module_name in sys.modules:
                del sys.modules[module_name]

            module = importlib.import_module(module_name)
            test_func = getattr(module, test_function_name)

            # Capture stdout
            from io import StringIO
            import contextlib

            temp_out = StringIO()
            with contextlib.redirect_stdout(temp_out):
                test_func()

            output = temp_out.getvalue()
            run_logs = parse_logs(output)
            logs.append(run_logs)
        except Exception as e:
            print(f"Run failed: {e}")

    write_logs_to_csv(logs, output_csv)

def parse_logs(output_str):
    # Find lines like "log>>> x: {...}"
    log_dict = {}
    for line in output_str.splitlines():
        if "log>>>" in line:
            match = re.match(r"log>>>\s*(\w+):\s*(.*)", line)
            if match:
                var_name, value = match.groups()
                log_dict[var_name] = value
    return log_dict

def write_logs_to_csv(logs, filename):
    if not logs:
        return

    fieldnames = set()
    for log in logs:
        fieldnames.update(log.keys())
    fieldnames = list(fieldnames)

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for log in logs:
            writer.writerow(log)

if __name__ == "__main__":
    # Example usage:
    # run_test_n_times('my_test_module', 'test_random_behavior', 100)
    pass
