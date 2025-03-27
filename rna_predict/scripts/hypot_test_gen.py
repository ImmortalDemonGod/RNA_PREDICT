import ast
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal, Any

import snoop  # type: ignore
# Removed unused: from hypothesis import strategies as st
import importlib.util  # For dynamic imports

# Set up logging with file and console output
log_file = "test_generator_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configure snoop to write to a separate debug log
snoop.install(out=Path("snoop_debug.log"))

def fix_leading_zeros(test_code: str) -> str:
    """
    Replace decimal integers with leading zeros (except a standalone "0") with their corrected form.
    For example, "007" becomes "7" and "-0123" becomes "-123".
    """
    import re
    # Use a regex with negative lookbehind and lookahead to match numbers that start with one or more zeros.
    # The pattern (?<!\d)(-?)0+(\d+)(?!\d) ensures that a minus sign is captured if present,
    # and that only isolated numbers are matched.
    fixed_code = re.sub(r'(?<!\d)(-?)0+(\d+)(?!\d)', lambda m: m.group(1) + str(int(m.group(2))), test_code)
    return fixed_code

def remove_logger_lines(text: str) -> str:
    """
    Remove extraneous logging lines from the generated test content.
    This function filters out:
      - Lines starting with a bracketed or non-bracketed timestamp (e.g. "[2025-3-27 14:55:48,330] ..." or "2025-03-27 14:55:48,330 - ...").
      - Lines containing known noisy substrings such as 'real_accelerator.py:' or 'Setting ds_accelerator to'.
    """
    import re
    lines = text.splitlines()
    filtered = []
    timestamp_pattern = re.compile(r'^\[?\d{4}-\d{1,2}-\d{1,2}')
    for line in lines:
        # Skip lines matching a leading timestamp
        if timestamp_pattern.match(line):
            continue
        # Skip lines containing known noisy substrings
        if 'real_accelerator.py:' in line or 'Setting ds_accelerator to' in line:
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()

@dataclass
class TestableEntity:
    """Represents a class, method, or function that can be tested"""
    name: str
    module_path: str
    entity_type: Literal['class', 'method', 'function', 'instance_method']  # More restrictive type
    parent_class: Optional[str] = None


def fix_pythonpath(file_path: Path) -> None:
    """Ensure the module being tested is in Python's path"""
    parent_dir = str(file_path.parent.absolute())
    add_to_sys_path(parent_dir, "parent directory")

    if "src" in file_path.parts:
        src_path = construct_src_path(file_path)
        add_to_sys_path(src_path, "src directory")


def add_to_sys_path(path: str, description: str) -> None:
    """Helper function to add a path to sys.path if not already present"""
    if path not in sys.path:
        sys.path.insert(0, path)
        logger.debug(f"Added {description} to sys.path: {path}")


def construct_src_path(file_path: Path) -> str:
    """Construct the src path from the file path"""
    src_index = file_path.parts.index("src")
    src_path = str(Path(*file_path.parts[: src_index + 1]).absolute())
    return src_path


class ModuleParser(ast.NodeVisitor):
    """AST-based parser for Python modules"""

    def __init__(self):
        self.entities: List[TestableEntity] = []
        self.current_class: Optional[str] = None
        self.class_bases: Dict[str, List[str]] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name.startswith("_"):
            return
        self.store_class_bases(node)
        self.add_class_entity(node)
        self.process_class_contents(node)

    def store_class_bases(self, node: ast.ClassDef) -> None:
        """Store base classes for inheritance checking"""
        bases = []
        for base in node.bases:
            base_name = self.get_base_name(base)
            if base_name:
                bases.append(base_name)
        self.class_bases[node.name] = bases
        logger.debug(f"Stored bases for class {node.name}: {bases}")

    def get_base_name(self, base: ast.AST) -> Optional[str]:
        """Retrieve the base class name from the AST node"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            if isinstance(base.value, ast.Name):
                return f"{base.value.id}.{base.attr}"
        return None

    def add_class_entity(self, node: ast.ClassDef) -> None:
        """Add the class itself to entities"""
        self.entities.append(TestableEntity(node.name, "", "class"))
        logger.debug(f"Added class entity: {node.name}")

    def process_class_contents(self, node: ast.ClassDef) -> None:
        """Process the contents of the class"""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        logger.debug(f"Processed contents of class {node.name}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("_"):
            return
        if self.current_class:
            self.process_method(node)
        else:
            self.add_function_entity(node)

    def process_method(self, node: ast.FunctionDef) -> None:
        """Process a method within a class"""
        if self.should_skip_method(node):
            return

        is_instance_method = self.determine_instance_method(node)
        entity_type = "instance_method" if is_instance_method else "method"

        # The method path should include the class
        method_name = f"{self.current_class}.{node.name}" if self.current_class else node.name

        self.entities.append(
            TestableEntity(
                name=node.name,
                module_path="",
                entity_type=entity_type,
                parent_class=self.current_class,
            )
        )
        logger.debug(
            f"Added {'instance_method' if is_instance_method else 'method'} entity: {method_name}"
        )

    def determine_instance_method(self, node: ast.FunctionDef) -> bool:
        """Determine if the method is an instance method"""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id in {"classmethod", "staticmethod"}:
                return False
        return True

    def should_skip_method(self, node: ast.FunctionDef) -> bool:
        """Determine if the method should be skipped based on inheritance or naming"""
        current_bases = self.class_bases.get(self.current_class, [])
        if any(base in {"NodeVisitor", "ast.NodeVisitor"} for base in current_bases):
            if node.name.startswith("visit_"):
                logger.debug(f"Skipping inherited visit method: {node.name}")
                return True
        if node.name in {"__init__", "__str__", "__repr__", "property"}:
            logger.debug(f"Skipping magic or property method: {node.name}")
            return True
        return False

    def add_function_entity(self, node: ast.FunctionDef) -> None:
        """Add a standalone function to entities"""
        self.entities.append(TestableEntity(node.name, "", "function"))
        logger.debug(f"Added function entity: {node.name}")


def debug_command_output(cmd: str, stdout: str, stderr: str, returncode: int) -> None:
    """Helper function to debug command execution"""
    logger.debug("Command execution details:")
    logger.debug(f"Command: {cmd}")
    logger.debug(f"Return code: {returncode}")
    logger.debug(f"stdout length: {len(stdout)}")
    logger.debug(f"stderr length: {len(stderr)}")
    logger.debug("First 1000 chars of stdout:")
    logger.debug(stdout[:1000])
    logger.debug("First 1000 chars of stderr:")
    logger.debug(stderr[:1000])


class TestFixer(ast.NodeTransformer):
    """AST transformer to fix duplicate self parameters"""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        seen_self = False
        new_args = []

        for arg in node.args.args:
            if arg.arg == 'self':
                if not seen_self:
                    seen_self = True
                    new_args.append(arg)
            else:
                new_args.append(arg)

        node.args.args = new_args
        return node


def fix_duplicate_self(test_content: str) -> Optional[str]:
    """
    Fix duplicate self parameters in test content.

    Args:
        test_content: String containing the test code

    Returns:
        Fixed test code string, or None if parsing fails
    """
    try:
        tree = ast.parse(test_content)

        fixer = TestFixer()
        fixed_tree = fixer.visit(tree)

        try:
            return ast.unparse(fixed_tree)
        except AttributeError:
            import astunparse
            return astunparse.unparse(fixed_tree)

    except Exception as e:
        print(f"Error fixing test content: {e}")
        return None


class TestGenerator:
    """Manages generation of Hypothesis tests for Python modules"""

    def __init__(self, output_dir: Path = Path("generated_tests")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.verify_output_dir()

    def verify_output_dir(self) -> None:
        """Verify that the output directory exists and is writable"""
        logger.debug(f"Test generator initialized with output dir: {self.output_dir}")
        logger.debug(f"Output dir exists: {self.output_dir.exists()}")
        logger.debug(f"Output dir is writable: {os.access(self.output_dir, os.W_OK)}")

    def run_hypothesis_write(self, command: str) -> Optional[str]:
        """Execute hypothesis write command and return output if successful"""
        full_cmd = f"hypothesis write {command}"
        logger.debug(f"Executing hypothesis command: {full_cmd}")

        try:
            self.log_environment()
            env = self.prepare_environment()

            result = subprocess.run(
                full_cmd, shell=True, capture_output=True, text=True, env=env
            )

            debug_command_output(
                full_cmd, result.stdout, result.stderr, result.returncode
            )

            return self.process_hypothesis_result(result)

        except Exception as e:
            logger.error(f"Error running hypothesis: {e}", exc_info=True)
            return None

    def log_environment(self) -> None:
        """Log the current environment settings"""
        logger.debug(f"PYTHONPATH before modification: {os.getenv('PYTHONPATH')}")
        logger.debug(f"sys.path: {sys.path}")
        logger.debug(f"Current working directory: {os.getcwd()}")

    def prepare_environment(self) -> Dict[str, str]:
        """Prepare the environment variables for subprocess"""
        env = os.environ.copy()
        env["PYTHONPATH"] = ":".join(sys.path)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def process_hypothesis_result(self, result: subprocess.CompletedProcess) -> Optional[str]:
        """Process the result of the hypothesis command"""
        if result.returncode == 0 and result.stdout:
            content = result.stdout.strip()
        
            # Remove extraneous logging lines first
            content = remove_logger_lines(content)
        
            if not content or len(content) < 50:
                logger.warning("Hypothesis generated insufficient content")
                return None
        
            # Process and fix the test content using post_process_test_content
            fixed_content = self.post_process_test_content(content)
            if fixed_content is None:
                logger.warning("Failed to process test content")
                return None

            logger.info("Successfully generated and processed test content")
            return fixed_content

        if result.stderr and not self.is_known_error(result.stderr):
            logger.warning(f"Command failed: {result.stderr}")
        return None

    def post_process_test_content(self, content: str) -> Optional[str]:
        """Post-process generated test content"""
        try:
            # Also remove extraneous logger lines (defensive)
            content = remove_logger_lines(content)
    
            # First, fix any leading zeros in integer literals
            content = fix_leading_zeros(content)
            # Then, fix duplicate self parameters
            fixed_content = fix_duplicate_self(content)
            if fixed_content is None:
                logger.warning("Failed to fix duplicate self parameters.")
                return content
            return fixed_content
        except Exception as e:
            logger.error(f"Error processing test content: {e}", exc_info=True)
            return None

    def is_known_error(self, stderr: str) -> bool:
        """Check if the stderr contains known non-critical errors"""
        known_errors = [
            "InvalidArgument: Got non-callable",
            "Could not resolve",
            "but it doesn't have a",
        ]
        return any(msg in stderr for msg in known_errors)

    def try_generate_test(
        self, entity: TestableEntity, variant: Dict[str, str], max_retries: int = 3
    ) -> bool:
        """Attempt to generate a specific test variant with retries"""
        for attempt in range(1, max_retries + 1):
            logger.debug(
                f"Attempt {attempt} for {variant['type']} test on {entity.name}"
            )
            output = self.attempt_test_generation(entity, variant, attempt)
            if output:
                return True
        return False

    def attempt_test_generation(
        self, entity: TestableEntity, variant: Dict[str, str], attempt: int
    ) -> Optional[bool]:
        """Attempt a single test generation"""
        output = self.run_hypothesis_write(variant["cmd"])
        if output:
            return self.handle_generated_output(entity, variant, output)
        else:
            return self.handle_failed_attempt(entity, variant, attempt)

    def handle_generated_output(
        self, entity: TestableEntity, variant: Dict[str, str], output: str
    ) -> bool:
        """Handle the output from a successful hypothesis generation"""
        name_prefix = (
            f"{entity.parent_class}_{entity.name}"
            if entity.parent_class
            else entity.name
        )
        output_file = self.output_dir / f"test_{name_prefix}_{variant['type']}.py"

        try:
            self.write_and_verify_output(output_file, output)
            logger.info(f"Successfully generated test at {output_file}")
            print(f"Generated {variant['type']} test: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error writing test file: {e}", exc_info=True)
            return False

    def write_and_verify_output(self, output_file: Path, content: str) -> None:
        """Write the test content to a file and verify its integrity"""
        logger.debug("Test content details:")
        logger.debug(f"Content length: {len(content)}")
        logger.debug(f"Content preview:\n{content[:1000]}")
        logger.debug(f"Writing to file: {output_file}")

        output_file.write_text(content)

        written_content = output_file.read_text()
        if not written_content:
            logger.error(f"File {output_file} is empty after writing!")
            raise ValueError(f"Empty file: {output_file}")

        if written_content != content:
            logger.error("Written content doesn't match original content!")
            logger.debug(f"Original length: {len(content)}")
            logger.debug(f"Written length: {len(written_content)}")
            raise ValueError("Content mismatch after writing")

        logger.debug(f"Final file size: {output_file.stat().st_size} bytes")

    def handle_failed_attempt(
        self, entity: TestableEntity, variant: Dict[str, str], attempt: int
    ) -> Optional[bool]:
        """Handle a failed test generation attempt"""
        if attempt < 3:
            logger.warning(f"Attempt {attempt} failed, retrying...")
            time.sleep(1)
        else:
            logger.error(f"All attempts failed for {entity.name}")
        return None

    def get_module_contents(self, file_path: Path) -> Tuple[str, List[TestableEntity]]:
        """Extract module path and testable entities using AST parsing"""
        logger.debug(f"Reading file: {file_path}")
        try:
            module_path = self.construct_module_path(file_path)
            content = file_path.read_text()
            parser = self.parse_ast(content)
            imports = self.extract_imports(content)

            entities = self.populate_entities(parser, module_path)
            self.log_entities_summary(entities)
            return module_path, entities

        except Exception as e:
            logger.error(f"Error parsing module contents: {e}", exc_info=True)
            raise

    def construct_module_path(self, file_path: Path) -> str:
        """Construct the module path from the file path"""
        parts = file_path.parts
        if "src" in parts:
            src_index = parts.index("src")
            module_parts = list(parts[src_index + 1 :])
        else:
            module_parts = [file_path.stem]
        module_path = ".".join([p.replace(".py", "") for p in module_parts])
        logger.debug(f"Constructed module path: {module_path}")
        return module_path

    def parse_ast(self, content: str) -> ModuleParser:
        """Parse the AST of the given content"""
        tree = ast.parse(content)
        parser = ModuleParser()
        parser.visit(tree)
        return parser

    def extract_imports(self, content: str) -> set:
        """Extract import statements from the content"""
        tree = ast.parse(content)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        logger.debug(f"Found imports: {imports}")
        return imports

    def populate_entities(self, parser: ModuleParser, module_path: str) -> List[TestableEntity]:
        """Populate entities with correct module paths"""
        entities = []
        for entity in parser.entities:
            entity.module_path = module_path
            entities.append(entity)
        return entities

    def log_entities_summary(self, entities: List[TestableEntity]) -> None:
        """Log a summary of found entities"""
        classes = sum(1 for e in entities if e.entity_type == "class")
        methods = sum(
            1 for e in entities if e.entity_type in {"method", "instance_method"}
        )
        functions = sum(1 for e in entities if e.entity_type == "function")
        logger.info(
            f"Found {classes} classes, {methods} methods, and {functions} functions"
        )

    def generate_all_tests(self, file_path: Path) -> None:
        """Generate all possible test variants for a Python file"""
        logger.info(f"Generating tests for file: {file_path}")
        try:
            fix_pythonpath(file_path)
            module_path, entities = self.get_module_contents(file_path)
            self.display_module_info(module_path, entities)
            total_variants = sum(len(self.generate_test_variants(e)) for e in entities)
            self.process_entities(entities, total_variants, module_path)
            print()
        except Exception:
            logger.error("Test generation failed", exc_info=True)
            raise

    def display_module_info(self, module_path: str, entities: List[TestableEntity]) -> None:
        """Display information about the module and its entities"""
        print(f"\nProcessing module: {module_path}")
        print(
            f"Found {len([e for e in entities if e.entity_type == 'class'])} classes, "
            f"{len([e for e in entities if e.entity_type in {'method', 'instance_method'}])} methods, and "
            f"{len([e for e in entities if e.entity_type == 'function'])} functions"
        )

    def process_entities(self, entities: List[TestableEntity], total_variants: int, module_path: str) -> None:
        """Process each entity and generate tests"""
        current = 0
        for entity in entities:
            print(f"\nGenerating tests for: {module_path}.{entity.name}")
            variants = self.generate_test_variants(entity)
            for variant in variants:
                current += 1
                print(f"\rGenerating tests: [{current}/{total_variants}]", end="")
                self.try_generate_test(entity, variant)
        print()

    def _get_object(self, path: str) -> Optional[Any]:
        """Get the actual object from its module path"""
        try:
            module_parts = path.split('.')
            module_path = '.'.join(module_parts[:-1])
            obj_name = module_parts[-1]

            spec = importlib.util.find_spec(module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, obj_name, None)
        except Exception:
            return None

    def generate_method_variants(self, entity: TestableEntity) -> List[Dict[str, str]]:
        """Generate test variants for methods and instance methods"""
        if entity.entity_type in {"method", "instance_method"}:
            method_path = f"{entity.module_path}.{entity.parent_class}.{entity.name}"
        else:
            method_path = f"{entity.module_path}.{entity.name}"

        # Start with basic test with type inference
        variants = [
            self.create_variant(
                "basic",
                f"--style=unittest --annotate {method_path}"
            )
        ]

        # Add error variant
        variants.append(
            self.create_variant(
                "errors",
                f"--style=unittest --annotate --except ValueError --except TypeError {method_path}"
            )
        )

        # Add special variants based on method name
        name = entity.name.lower()
        variants.extend(self._generate_special_variants(name, method_path))

        return variants

    def _generate_special_variants(self, name: str, method_path: str) -> List[Dict[str, str]]:
        """Generate special variants based on method name"""
        special_variants = []

        if any(x in name for x in ["transform", "convert", "process", "format"]):
            special_variants.append(
                self.create_variant(
                    "idempotent",
                    f"--style=unittest --annotate --idempotent {method_path}"
                )
            )

        if any(x in name for x in ["validate", "verify", "check", "assert"]):
            special_variants.append(
                self.create_variant(
                    "validation",
                    f"--style=unittest --annotate --errors-equivalent {method_path}"
                )
            )

        if "encode" in name or "decode" in name:
            special_variants.append(
                self.create_variant(
                    "roundtrip",
                    f"--style=unittest --annotate --roundtrip {method_path}"
                )
            )

        if any(x in name for x in ["add", "multiply", "subtract", "combine", "merge"]):
            special_variants.append(
                self.create_variant(
                    "binary-op",
                    f"--style=unittest --annotate --binary-op {method_path}"
                )
            )

        return special_variants

    def generate_function_variants(self, entity: TestableEntity) -> List[Dict[str, str]]:
        """Generate test variants for standalone functions"""
        base_cmd = f"--style=unittest --annotate {entity.module_path}.{entity.name}"
        variants = [self.create_variant("basic", base_cmd)]

        # Add special variants for functions if needed
        name = entity.name.lower()
        if "encode" in name or "decode" in name or "serialize" in name or "deserialize" in name:
            variants.append(self.create_variant("roundtrip", f"{base_cmd} --roundtrip"))
        elif any(x in name for x in ["add", "sub", "mul", "combine", "merge"]):
            variants.append(self.create_variant("binary-op", f"{base_cmd} --binary-op"))

        return variants

    def generate_test_variants(self, entity: TestableEntity) -> List[Dict[str, str]]:
        """Generate all applicable test variants for an entity"""
        variants = []
        if entity.entity_type == "class":
            # For classes, just a basic annotated variant
            variants.append(self.create_variant("basic", f"--style=unittest --annotate {entity.module_path}.{entity.name}"))
        elif entity.entity_type in {"method", "instance_method"}:
            variants.extend(self.generate_method_variants(entity))
        else:
            variants.extend(self.generate_function_variants(entity))
        logger.debug(f"Generated variants for {entity.name}: {[v['type'] for v in variants]}")
        return variants

    def create_variant(self, variant_type: str, cmd: str) -> Dict[str, str]:
        """Create a test variant dictionary with properly formatted command"""
        return {
            "type": variant_type,
            "cmd": cmd.strip()  # Ensure no extra whitespace in command
        }


def parse_args(args: Optional[list] = None) -> Path:
    """
    Parse command line arguments and validate file path.

    Args:
        args: Optional list of command line arguments. If None, uses sys.argv[1:]

    Returns:
        Path object for the input file

    Raises:
        ValueError: If arguments are invalid or file doesn't exist
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) != 1:
        raise ValueError("Exactly one argument (path to Python file) required")

    file_path = Path(args[0])
    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"File does not exist or is not a file: {file_path}")

    return file_path


def run_test_generation(file_path: Union[str, Path]) -> bool:
    """
    Run the test generation process for a given file.

    Args:
        file_path: Path to the Python file to generate tests for

    Returns:
        bool: True if test generation was successful, False otherwise

    Raises:
        Exception: If test generation fails
    """
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        logger.info(f"Starting test generator for {file_path}")
        generator = TestGenerator()
        generator.generate_all_tests(file_path)
        return True

    except Exception as e:
        logger.error(f"Test generation failed: {e}", exc_info=True)
        return False


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the test generator script.

    Args:
        args: Optional list of command line arguments. If None, uses sys.argv[1:]

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        file_path = parse_args(args)
        success = run_test_generation(file_path)
        return 0 if success else 1

    except ValueError as e:
        print(f"Error: {e}")
        logger.error(f"Invalid arguments: {e}")
        print("Usage: python test_generator.py <path_to_python_file>")
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}")
        logger.error("Unexpected error during execution", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())