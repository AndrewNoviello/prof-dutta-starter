import os
import ast
import csv

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


# Returns inheritance class graph (should be a DAG) adjacency list.
def build_inheritance_graph(classes):
    inheritance_graph = {}

    for class_name, class_node in classes.items():
        base_names = extract_base_class_names(class_node)
        inheritance_graph[class_name] = base_names

    return inheritance_graph

# Reverses all the edges of the inheritance graph DAG. This is so that
# we can look up from a subclass to find all its superclasses that it inherits from.
def reverse_inheritance_graph(classes):
    # First build the normal inheritance graph
    inheritance_graph = build_inheritance_graph(classes)

    # Initialize the reverse graph with empty lists for all classes
    reverse_graph = {class_name: [] for class_name in classes.keys()}

    # For each class, add it as a subclass to all its base classes
    for class_name, base_classes in inheritance_graph.items():
        for base_class in base_classes:
            # Make sure the base class exists in the reverse graph
            if base_class in reverse_graph:
                reverse_graph[base_class].append(class_name)
            else:
                # Handle case where a base class isn't in the original classes dict
                reverse_graph[base_class] = [class_name]

    return reverse_graph

# Check if an expression is a constant float. Returns true if float, false otherwise.
# Note that in python, variable types are obviously not kept in the AST because
# they are determined at run-time due to dynamic typing. As such, we have no way
# of knowing whether a Name (variable) expression is a float or not.
def is_float(expr):
    # An expression is only a float if it is both a Constant and a float type
    return isinstance(expr, ast.Constant) and isinstance(expr.value, float)

# Check if a comparison operator could be flaky. I assume that comparisons are only
# flaky if either the left or right side is a float.
def is_flaky_compare(node):
    if isinstance(node, ast.Compare):
        return any(isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)) for op in node.ops) and \
            (is_float(node.left) or any(is_float(comp) for comp in node.comparators))
    return False

# Extract the assertion type
def extract_assertion_type(assert_string):
    for assertion_type in always_flaky_assertion_types + check_flaky_assertion_types:
        if assertion_type in assert_string:
            return assertion_type
    return "other"

# Find approximate assertions in a function
def find_approximate_assertions(func_node, source_code):
    """Find approximate assertions inside a function, treating always and sometimes flaky assertions separately."""
    assertions = []
    for stmt in ast.walk(func_node):
        if isinstance(stmt, ast.Call):
            if isinstance(stmt.func, ast.Name):
                name = stmt.func.id
            elif isinstance(stmt.func, ast.Attribute):
                name = stmt.func.attr
            else:
                continue
            # Handle always-flaky assertions
            if name in always_flaky_assertion_types:
                assertions.append((stmt.lineno, ast.get_source_segment(source_code, stmt)))
            # Handle sometimes-flaky assertions
            elif name in check_flaky_assertion_types and is_flaky_compare(stmt):
                assertions.append((stmt.lineno, ast.get_source_segment(source_code, stmt)))
    return assertions

# Recursively trace approximate assertions through call graph
def trace_assertions(start_func, functions, call_graph, source_code, memo=None):
    if memo is None:
        memo = set()

    results = []
    if start_func in memo:
        return results
    memo.add(start_func)

    if start_func not in functions:
        return results

    func_node = functions[start_func]
    results.extend(find_approximate_assertions(func_node, source_code))

    for callee in call_graph.get(start_func, []):
        resolved = resolve_call(start_func, callee, functions)
        if resolved:
            results.extend(trace_assertions(resolved, functions, call_graph, source_code, memo))

    return results

# Extract base class names for inheritance handling
def extract_base_class_names(class_node):
    base_names = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
    return base_names

# Process a single test file and write findings to CSV.
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
                    class_name = ''
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

# Run the main driver on all of the projects
main('./projects/magenta', 'magenta')
main('./projects/botorch', 'botorch')
main('./projects/sonnet', 'sonnet')

