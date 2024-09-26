import pylint.lint
from pylint.reporters.text import TextReporter
from io import StringIO
import os

def lint_code_snippet(code_snippet):
    """Lints a code snippet using pylint."""

    temp_file_path = 'temp.py'
    # Create a temporary file to write the code snippet to, as PyLinter requires file paths
    with open(temp_file_path, 'w') as temp_file:
        temp_file.write(code_snippet)
     
    # Setup the in-memory output stream for pylint reports
    output = StringIO()
    
    # Initialize the linter
    linter = pylint.lint.PyLinter(reporter=TextReporter(output=output))
    linter.load_default_plugins()  # Load the default pylint plugins
    linter.check([temp_file_path])
    
    os.remove(temp_file_path)
    # Return the captured output
    return output.getvalue()

# Your code snippet (unchanged)
code_snippet = """
def hello_world():
    print("Hello, World!")

hello_world()
"""
lint_output = lint_code_snippet(code_snippet)
print(lint_output)