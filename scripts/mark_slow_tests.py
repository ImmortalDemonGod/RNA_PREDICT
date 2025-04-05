#!/usr/bin/env python3
import os
import re
import ast
from pathlib import Path

SLOW_TEST_THRESHOLD = 1.0  # seconds

def find_test_files(root_dir):
    """Find all test files in the given directory."""
    test_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    return test_files

def parse_test_file(file_path):
    """Parse a test file and return its AST."""
    with open(file_path, 'r') as f:
        return ast.parse(f.read())

def is_test_function(node):
    """Check if a node is a test function."""
    return (isinstance(node, ast.FunctionDef) and 
            node.name.startswith('test_'))

def add_slow_marker(file_path):
    """Add @pytest.mark.slow decorator to slow test functions."""
    tree = parse_test_file(file_path)
    modified = False
    
    class SlowTestMarker(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if is_test_function(node):
                # Check if the test is already marked as slow
                has_slow_marker = any(
                    isinstance(d, ast.Call) and 
                    isinstance(d.func, ast.Attribute) and
                    d.func.attr == 'mark' and
                    isinstance(d.func.value, ast.Attribute) and
                    d.func.value.attr == 'pytest' and
                    any(kw.arg == 'slow' for kw in d.keywords)
                    for d in node.decorator_list
                )
                
                if not has_slow_marker:
                    # Add the slow marker
                    slow_marker = ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id='pytest', ctx=ast.Load()),
                                attr='mark',
                                ctx=ast.Load()
                            ),
                            attr='slow',
                            ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[]
                    )
                    node.decorator_list.insert(0, slow_marker)
                    nonlocal modified
                    modified = True
            return node

    if modified:
        tree = SlowTestMarker().visit(tree)
        with open(file_path, 'w') as f:
            f.write(ast.unparse(tree))
        print(f"Added slow marker to tests in {file_path}")

def main():
    """Main function to mark slow tests."""
    root_dir = Path('tests')
    test_files = find_test_files(root_dir)
    
    for file_path in test_files:
        add_slow_marker(file_path)

if __name__ == '__main__':
    main() 