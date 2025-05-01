import ast
import csv
import re
from io import StringIO
import contextlib
import os

# Adds the logging function to the top of the AST tree.
# This logging function will handle any datatype and print
# the object accordingly
def add_logging_function(tree):
    # Note that the top condition holds for numpy arrays and pytorch tensors
    code_for_log_function = """
def safe_log(x):
    try:
        if hasattr(x, 'shape') or hasattr(x, '__len__'):
            return {'min': x.min(), 'max': x.max()}
        else:
            return x
    except Exception as e:
        return ""
"""
    safe_log_ast = ast.parse(code_for_log_function)
    tree.body = safe_log_ast.body + tree.body
    return tree

# This function generates the AST format for the logging, using
# the safe_log function defined below. We will insert this response
# into the AST above the assert statements.
def generate_logging_code(var_name):
    return [
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[
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

# Generates import statements for the AST tree
def generate_import_nodes():
    return [
        ast.Import(names=[ast.alias(name='random', asname=None)]),
        ast.Import(names=[ast.alias(name='numpy', asname='np')]),
        ast.Import(names=[ast.alias(name='torch', asname=None)]),
        ast.Import(names=[ast.alias(name='tensorflow', asname='tf')]),
        ast.Import(names=[ast.alias(name='tree', asname=None)])
    ]

# Returns the ast syntax for the random seed generation.
def generate_seed_code(seed=42):
    # Sets the following seeds to cover all the cases:
    # Numpy random.seed
    # TensorFlow random.set_seed
    # TensorFlow set_random_seed
    # TensorFlow random.set_random_seed
    # TensorFlow compat.v1.random.set_random_seed
    # PyTorch manual_seed
    # PyTorch cuda.manual_seed_all
    # PyTorch seed
    # Random (Python) seed
    return [
        ast.parse(f"np.random.seed({seed})").body[0],
        ast.parse(f"tf.random.set_seed({seed})").body[0],
        # ast.parse(f"tf.set_random_seed({seed})").body[0],
        # ast.parse(f"tf.random.set_random_seed({seed})").body[0],
        ast.parse(f"tf.compat.v1.random.set_random_seed({seed})").body[0],
        ast.parse(f"torch.manual_seed({seed})").body[0],
        ast.parse(f"torch.cuda.manual_seed_all({seed})").body[0],
        # ast.parse(f"torch.seed({seed})").body[0],
        ast.parse(f"random.seed({seed})").body[0]
    ]


# Extracts variables from an assert expression
# AST object. Returns list of string variable names.
# Note that this is used for vanilla assert statements
# and for function call assertions.
def extract_assert_vars(expr):
    try:
        return [ast.unparse(expr)]
    except Exception:
        return []

# Main instrumentor function. Returns the instrumented source code.
def instrumentor(source, test_name, assertion_line):
    # Parse the ast tree from the source code
    tree = ast.parse(source)

    # Add the logging function to the ast tree
    tree = add_logging_function(tree)

    instrumented = False

    # Walk the full nodes of the tree
    for function_node in ast.walk(tree):
        # Check if we have found a function named the same as our test.
        # Note that we are not going to worry about full scope paths for the
        # instrumentor as we are only doing this for a small number of tests
        # and the odds that there is a scope conflict is quite small.
        if isinstance(function_node, ast.FunctionDef) and function_node.name == test_name:
            # Insert random seeds at start of test
            function_node.body = generate_seed_code() + function_node.body
            # Loop through all the statements in the function. This should include our
            # assert statements. Thus, the instrumentor will only work for top-level
            # assertions, not the nested ones. This is fine because we only need to run
            # it on 10 tests for the projects and there are certainly well over 10 top-level
            # tests in each project (most of the tests in general are top-level).
            for stmt_node in ast.walk(function_node):
                # Skip if not same line number as out assertion
                if not hasattr(stmt_node, 'lineno') or stmt_node.lineno != int(assertion_line):
                    continue

                # Direct assert statement for expression, etc.
                if isinstance(stmt_node, ast.Assert):
                    targets = extract_assert_vars(stmt_node.test)
                    logging_nodes = []
                    for var in targets:
                        # Log this particular variable
                        logging_nodes += generate_logging_code(var)
                # Assertion function call for one of the libraries, etc.
                elif isinstance(stmt_node, ast.Expr) and isinstance(stmt_node.value, ast.Call):
                    # Same as above, but uses the function args instead of the
                    # compared variables directly.
                    call = stmt_node.value
                    targets = []
                    for arg in call.args:
                        targets += extract_assert_vars(arg)
                    logging_nodes = []
                    for var in targets:
                        logging_nodes += generate_logging_code(var)
                else:
                    continue

                # This loop is to find the parent node of the assertion,
                # so that we may insert the logging code appropriately directly above
                # the assertion.
                for parent in ast.walk(function_node):
                    # This node is in fact the parent of our assertion
                    if (hasattr(parent, 'body') and isinstance(parent.body, list) and
                        any(stmt is stmt_node for stmt in parent.body)):
                        # Loop through the parent body and insert the logging nodes
                        # directly above the assertion
                        for idx, stmt in enumerate(parent.body):
                            if stmt is stmt_node:
                                parent.body = parent.body[:idx] + logging_nodes + parent.body[idx:]
                                instrumented = True
                                break
                        break

    # Fix the tree and return the new source code
    if instrumented:
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    else:
        print(f"No matching assertion found. {test_name} at line {assertion_line}")
        return None

# Find a function by name in the AST. Again, we are not worried
# about scope path conflicts for the instrumentor project.
def find_function_ast_in_module(source_code, target_name):
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == target_name:
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef) and node in parent.body:
                    return node, parent.name  # method + class name
            return node, None  # function not in class
    return None, None

# Run the code from the ast. Main runner function.
def run(source_code, test_function_name, n_runs=100, output_dir="output"):
    # Parse the full source
    tree = ast.parse(source_code)

    # Add the necessary standard imports to the top of the tree body
    tree.body = generate_import_nodes() + tree.body

    # Fix the line numbers, etc. of the tree
    ast.fix_missing_locations(tree)

    # Note that I used Claude to help with this next bit. I had a very different
    # implementation at first, that needed to be revised.

    # Compile and execute the module code.
    # https://stackoverflow.com/questions/768634/parse-a-py-file-read-the-ast-modify-it-then-write-back-the-modified-source-c
    code = compile(tree, filename="<ast>", mode="exec")
    func_namespace = {}
    exec(code, func_namespace)

    # Locate the test function and class again in the module
    func_node, class_name = find_function_ast_in_module(source_code, test_function_name)

    # We need to instantiate the class if we have a method instead
    # of a top level function, so these need to be handled separately.
    if class_name:
        instance = func_namespace[class_name]()
        test_func = getattr(instance, test_function_name)
    else:
        test_func = func_namespace[test_function_name]

    all_outputs = []
    # Run the code n_runs times
    for i in range(n_runs):
        try:
            temp_out = StringIO()
            # Redirects the print IO to temp_out
            # Note to self: you cannot put a print statement in here,
            # it will be redirected to temp_out too!
            with contextlib.redirect_stdout(temp_out):
                test_func()
            all_outputs.append(temp_out.getvalue())
        except Exception as e:
            print(f"Run failed: {e}")

    # If there is content to write, write it to an output txt fule
    if len(all_outputs) > 0:
        with open(os.path.join(output_dir, f"output_{test_function_name}.txt"), "w", encoding="utf-8") as f:
            for i, output in enumerate(all_outputs):
                f.write(output)
                f.write("\n")

# Standard code to loop through the csv. Boiler-plate function mostly.
def process_csv_and_run(csv_path, base_dir="", proj="output", n_runs=100, max_tests=10):
    # Read the csv file, used the csv module instead of pandas, which would normally
    # be the go-to.
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        entries = list(reader)[:max_tests]

    # Loop through the assertions
    for idx, row in enumerate(entries):
        filepath = os.path.join(base_dir, row['filepath'])
        testname = row['testname'].strip()
        line_number = int(row['line number'])

        # Read the source code for the relevant filepath
        with open(filepath, "r", encoding="utf-8") as f:
            original_source = f.read()

        print(f"\n[{idx+1}] Instrumenting {filepath} at line {line_number} for test {testname}...")

        new_source = instrumentor(original_source, testname, line_number)
        if new_source is None:
            print("Instrumentation failed, skipping.")
            continue

        try:
            run(new_source, testname, n_runs=n_runs, output_dir=proj)
        except Exception as e:
            print(f"Failed to run instrumented test {testname}. Error: {e}")


# Note that this will only instrument assertions in the top-level
# of their functions. It will skip assertions in functions called by another function.
# This is a purposeful design decision to prevent double-instrumentation. In a real
# production setting, I could extend the existing code to that use-case, but I don't
# think it is really necessary for a quick starter task like this (seems over-kill to me).
# If you are reviewing this code, professor, and you want me to extend the instrumentation
# to cover such cases, just let me know via email and I will gladly do so, but I figured
# you probably wouldn't care. Thanks!
if __name__ == "__main__":
    # Example usage
    process_csv_and_run(
        csv_path="../task2results/sonnet_assertions.csv",  # path to your CSV
        base_dir="../",                  # where the python files are relative to
        proj="sonnet",
        n_runs=100,
        max_tests=1000
    )