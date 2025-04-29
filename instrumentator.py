import ast
import astor
import random

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
            # If it's an array/tensor-like object
            return {'min': x.min(), 'max': x.max()}
        else:
            return x
    except Exception as e:
        return str(x)
"""
    safe_log_ast = ast.parse(safe_log_code)
    tree.body = safe_log_ast.body + tree.body
    return tree

# Main function to instrument a file
def instrumentor(file_path, test_name, assertion_line):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    tree = insert_safe_log_function(tree)  # Insert the helper function at top

    modified = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            # Insert random seeds at start of test
            seed_code = generate_seed_code()
            node.body = seed_code + node.body

            for idx, stmt in enumerate(node.body):
                # Look for assertion at correct line
                if hasattr(stmt, 'lineno') and stmt.lineno == int(assertion_line):
                    if isinstance(stmt, ast.Assert):
                        # Try to extract left and right sides if it's a comparison
                        targets = extract_assert_vars(stmt.test)
                        logging_nodes = []
                        for var in targets:
                            logging_nodes += generate_logging_code(var)
                        node.body = node.body[:idx] + logging_nodes + node.body[idx:]
                        modified = True
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        # Handles things like self.assertTrue(x < y)
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
        new_source = astor.to_source(tree)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_source)
    else:
        print(f"No matching assertion found in {file_path} at line {assertion_line}")

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

