import os
import ast
import csv
import sys

always_flaky_assertion_types = ['assert_allclose', 'assert_almost_equal', 'assert_approx_equal',
                                'assert_array_almost_equal', 'assert_array_less', 'assertAllClose']

check_flaky_assertion_types = ['assertTrue', 'assertFalse', 'assertGreater', 'assertGreaterEqual',
                               'assertLess', 'assertLessEqual']

# Fetch the files that we want from the root directory of the relevant project.
# Returns the full file paths from the root.
def fetch_files(root_dir):
    return [os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(root_dir)
            for filename in filenames if 'test' in filename and filename.endswith('.py')]

# Parse the AST from a file. Note that we only need to consider the AST locally for
# this task, as mentioned by the instructions.
def read_and_parse(file):
    with open(file, 'r', encoding='utf-8') as f:
        source = f.read()
    return ast.parse(source, filename=file), source

# Parse all the AST trees from a list of files. Returns a dictionary, mapping
# file path to AST tree. This mapping is necessary for the dependency graph.
def parse_ast_trees(files):
    return {file: read_and_parse(file)[0] for file in files}

# In order to build the call graph, we need to walk through the entire tree.
# We cannot just use the ast.walk because we want to get the full scope path
# for all the functions to ensure that we do not make incorrect assumptions in tracing.
def collect_definitions(tree):
    functions = {}
    classes = {}

    def visit(node, path):
        # If the current node is a function
        if isinstance(node, ast.FunctionDef):
            full_name = '.'.join(path + [node.name])
            functions[full_name] = node
            # Visit all the child nodes of a function
            for child in node.body:
                visit(child, path + [node.name])
        # If the current node is a class
        elif isinstance(node, ast.ClassDef):
            full_name = '.'.join(path + [node.name])
            classes[full_name] = node
            # Visit all the child nodes of the class
            for child in node.body:
                visit(child, path + [node.name + "_class"])
                # We append a "_class" to mark this part of the scope as a class. This will
                # be used later when we are tracing inheritance.
        # If the current node is something else (a conditional, loop, etc.),
        # we want to check for function/class definitions inside of them
        elif hasattr(node, 'body') and isinstance(node.body, list):
            for child in node.body:
                visit(child, path)

    visit(tree, [])
    return functions, classes


# This function builds a call graph from the each of the functions, using
# proper scoping. Returns an adjacency list object to represent the graph.
# The graph is directed, but not acyclic due to the possibility
# of mutual recursion. Note that we could even extend this to work for
# imported functions, but that is not in the spec of this starter task.
def build_call_graph(functions):
    call_graph = {}

    # Loop through the provided functions, with the full scoped function names and
    # their associated nodes.
    for func_name, func_node in functions.items():
        # Functions called by the current function.
        called = []

        # Walk through all the statements in the function node.
        for stmt in ast.walk(func_node):
            if isinstance(stmt, ast.Call):
                # For a standalone function call
                if isinstance(stmt.func, ast.Name):
                    called.append(stmt.func.id)
                # For a static, class, or object method call
                elif isinstance(stmt.func, ast.Attribute):
                    called.append(stmt.func.attr)

        call_graph[func_name] = called

    return call_graph

# This function takes in the current scope path and the
# the name of the function being called and determines which
# function it is with full scope
def resolve_call(current_path, called_name, functions):
    # Check each scope, from nearest upward
    scopes = current_path.split('.')
    while scopes:
        possible = '.'.join(scopes + [called_name])
        if possible in functions:
            return possible
        scopes.pop()
    # Top-level functions fallback
    # Note that if we were to extend this to work across files,
    # this would be where we would check imported functions too.
    if called_name in functions:
        return called_name
    return None

# This function takes in the current scope path and the
# the name of the superclass being inherited and determines which
# class it is with full scope
def resolve_superclass(current_path, superclass_name, classes):
    # Check each scope, from nearest upward
    scopes = current_path.split('.')
    while scopes:
        possible = '.'.join(scopes + [superclass_name])
        if possible in classes:
            return possible
        scopes.pop()
    # Top-level functions fallback
    # Note that if we were to extend this to work across files,
    # this would be where we would check imported functions too.
    if superclass_name in classes:
        return superclass_name
    return None

# Build the reverse inheritance graph. This is a DAG because there is not,
# to my knowledge, any form of circularity in inheritance in python.
def reverse_inheritance_graph(classes):
    reverse_graph = {class_name: [] for class_name in classes.keys()}

    for class_name, class_node in classes.items():
        for base in class_node.bases:
            # We are not going to consider pathological inheritance classes
            # For instance, inheriting from nested classes. This just gets silly
            # and I don't really want to deal with it, especially I think it
            # is highly unlikely that this is exhibited in any of the project.
            if isinstance(base, ast.Name):
                base_name = resolve_superclass(class_name, base.id, classes)
                if base_name:
                    reverse_graph[class_name].append(base_name)

    return reverse_graph

# Check if an expression is a constant float. Returns true if float, false otherwise.
# Note that in python, variable types are obviously not kept in the AST because
# they are determined at run-time due to dynamic typing. As such, we have no way
# of knowing whether a Name (variable) expression is a float or not.
def is_float(expr):
    # An expression is only a float if it is both a Constant and a float type
    return isinstance(expr, ast.Constant) and isinstance(expr.value, float)

# Check if a comparison operator could be flaky. I assume that comparisons are only
# flaky if either the left or right side is a constant float. This obviously isn't
# comprehensive (e.g. two variable floats), but I wasn't sure how to handle that
# case with static analysis alone.
def is_flaky_compare(node):
    if isinstance(node, ast.Compare):
        return any(isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)) for op in node.ops) and \
            (is_float(node.left) or any(is_float(comp) for comp in node.comparators))
    return False

# Get the assertion type based on the table mentioned in the project description.
# If the assertion is something different, we assume it is just a normal assertion
# based on conditions.
def extract_assertion_type(assert_string):
    for assertion_type in always_flaky_assertion_types + check_flaky_assertion_types:
        if assertion_type in assert_string:
            return assertion_type
    return "assert expr"

# Given a function node and the source code, find all the
# possible flaky tests.
def find_flaky_tests(func_node, source_code):
    assertions = []

    def visit(node):
        # If it's a new function or class, skip walking into it. These will be cataloged
        # separately and explored for assertions on their own.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = None

            if name in always_flaky_assertion_types:
                assertions.append((node.lineno, ast.get_source_segment(source_code, node)))
            elif name in check_flaky_assertion_types and is_flaky_compare(node):
                assertions.append((node.lineno, ast.get_source_segment(source_code, node)))

        # Recurse into children if it is not a new function or class.
        for child in ast.iter_child_nodes(node):
            visit(child)

    # Loop through all elements of the function body and start visiting from them.
    for stmt in func_node.body:
        visit(stmt)

    return assertions


# Traces assertions through the call graph recursively. Returns list of assertions.
def trace_assertions(func_name, functions, call_graph, source_code, visited=None):
    if visited is None:
        visited = set()

    # Recursive base case, stop if we've already visited this function or
    # it is not in the list of cataloged functions
    if func_name in visited or func_name not in functions:
        return []

    # Add the function full scope path name to the visited set
    visited.add(func_name)

    # Catalog the assertions in the current functions level
    results = find_flaky_tests(functions[func_name], source_code)

    # Loop through all the functions that this function calls based on the call graph
    for callee in call_graph.get(func_name, []):
        # Resolve which function full scope path it is calling
        resolved = resolve_call(func_name, callee, functions)
        if resolved:
            # Recursively extend the results by drilling down into the called functions
            results.extend(trace_assertions(resolved, functions, call_graph, source_code, visited))

    return results

# Process a single test file and write assertions to CSV
def process_file(filepath, writer):
    # Parse the ast tree and fetch the source code
    tree, source_code = read_and_parse(filepath)

    # Get all the class and function definitions
    functions, classes = collect_definitions(tree)

    # Build the call_graph
    call_graph = build_call_graph(functions)

    inheritance_graph = reverse_inheritance_graph(classes)

    # Mapping of functions (by full scope path) to their list of assertions
    func_assertion_mappings = {}

    # Loop through all the functions that we recorded
    for func_name, func_node in functions.items():
        # If the function is a 'root' function, a test itself
        if 'test' in func_name.split('.')[-1]:
            # Trace the assertions from this function through the call graph.
            assertions = trace_assertions(func_name, functions, call_graph, source_code)
            func_assertion_mappings[func_name] = []

            # Loop through the assertions that we found
            for lineno, assert_string in assertions:
                parts = func_name.split('.')
                method_name = parts[-1]
                if len(parts) >= 2 and "_class" in parts[-2]:
                    class_name = parts[-2]
                else:
                    class_name = ""
                func_assertion_mappings[func_name].append([filepath, class_name, method_name, extract_assertion_type(assert_string), lineno, assert_string])

    inheritance_assertions = []
    # Loop through all the classes
    for class_name, class_node in classes.items():
        superclasses = inheritance_graph[class_name]

        # Loop through all the functions in the relevant class
        for func_name, func_node in functions.items():
            parts = func_name.split('.')
            # This function is an immediate method of a superclass of our function.
            if len(parts) >= 2 and any(parts[-2] == f"{cn}_class" for cn in superclasses) and func_name in func_assertion_mappings:
                # We add all the same assertions from the parent class function,
                # only changing the class name for the child class.
                for assertion in func_assertion_mappings[func_name]:
                    assertion[1] = class_name
                    inheritance_assertions.append(assertion)

    # Now write all the assertions to the csv file
    for func in func_assertion_mappings:
        for assertion in func_assertion_mappings[func]:
            writer.writerow(assertion)
    for assertion in inheritance_assertions:
        writer.writerow(assertion)

# Main driver function for the Task 2, AssertSpecFinder
def main(project_dir, project_name):
    # Create the output csv path
    output_csv = f"{project_name}_assertions.csv"

    # Create the output csv file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write the heading to the csv file
        writer.writerow(["filepath", "testclass", "testname", "assertion type", "line number", "assert string"])

        # Fetch the relevant files and process them one by one
        files = fetch_files(project_dir)
        for filepath in files:
            process_file(filepath, writer)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <project_name>")
        sys.exit(1)

    project_name = sys.argv[1]
    project_dir = f"../projects/{project_name}"  # assume project directory is same as project name
    main(project_dir, project_name)

# Run the main driver on all of the projects
# main('../projects/magenta', 'magenta')
# main('../projects/botorch', 'botorch')
# main('../projects/sonnet', 'sonnet')

