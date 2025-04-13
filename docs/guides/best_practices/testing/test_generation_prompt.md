
Prompt: Synthesized Test File Generation for High Coverage

⸻

Mission
Craft a comprehensive, maintainable Python test file covering at least 80% of the target module’s code. Use the specified generated_tests file as an instructional guide, without copying it verbatim. Integrate standard unit tests, property-based tests, mocks (if needed), and round-trip checks (if relevant), aiming to deliver clear, PEP 8-compliant code.

⸻

Context
You have a folder, /Users/tomriddle1/RNA_PREDICT/generated_tests/, containing multiple automatically generated instruction files. Each file details how one might build a suite of Python tests for a corresponding code module. However, these instruction files may be repetitive, incomplete, or contain extraneous content. Your job is to do the following for the current file being processed:
	1.	Parse & Interpret the Instructions
	•	Identify the primary module or functionality these instructions reference.
	•	Extract important details: function/class coverage goals, potential edge cases, or example test patterns.
	•	Avoid copying large blocks of text verbatim. Instead, translate the instructions into actionable test code concepts.
	2.	Design the Final Test File
	•	Write a single Python test module (e.g., test_<module_name>.py or similar).
	•	Use unittest or pytest (choose whichever best suits the instructions or your style).
	•	Include a logical structure (e.g., classes or fixtures) that organizes tests by functionality or feature area.
	3.	Key Testing Elements
a. Behavioral (Unit) Tests
	•	Thoroughly cover the public methods and classes of the target module: normal cases, boundary conditions, and anticipated error paths.
	•	Adhere to best practices: clear naming, setup/teardown or fixtures, PEP 8 compliance.
b. Property-Based Tests
	•	When the instructions mention fuzzing, or automatically generated inputs, incorporate hypothesis strategies to cover a wide input range.
	•	Deploy relevant hypothesis decorators and strategies (st.integers, st.floats, st.builds, etc.) to ensure robust coverage.
	•	Add at least one or two property-based tests that systematically probe edge cases.
c. Round-Trip & Integration Checks
	•	If the instructions reference transformations, parsing/serialization, or encode/decode logic, include a round-trip test verifying that data remains consistent.
	•	For multi-step processes, consider an integration-style test to confirm components work together as expected.
d. Mocking & Isolation
	•	For dependencies on external services or nondeterministic behavior (randomness, time, I/O), use unittest.mock (or equivalent) to isolate the logic under test.
	•	Confirm that tests remain fast, self-contained, and repeatable.
e. Coverage Emphasis
	•	Seek to exercise every relevant branch or path. Check that each public function/class is included in some test scenario.
	•	Write additional test cases for unusual inputs or corner cases to lift coverage over 80%.
	4.	Deliver a Polished Python Test File
	•	The final output is one complete .py test file, which can be run directly (e.g., via python -m unittest or pytest).
	•	Use concise, readable code that a new contributor could quickly follow or a seasoned developer would trust.
	•	Avoid duplicating large text from the generated instructions; rework them into a clear, consolidated test suite.

⸻

Constraints & Style
	•	Do not merely replicate the entire instructions from the generated tests. Instead, interpret them to produce a concise but comprehensive test file.
	•	Maintain standard Python coding style (PEP 8) and typical test naming conventions.
	•	Include docstrings or inline comments where beneficial for clarity, but keep them focused.
	•	Ensure the final test code stands on its own—no extra harness or extraneous scaffolding.

⸻

Your Output
	1.	A well-structured Python code block representing the final test file.
	2.	Immediately runnable under unittest or pytest.
	3.	Incorporates all test categories (behavioral, property-based, round-trip, mocking as needed), fulfilling the coverage objectives.
	4.	Does not quote or copy instruction text verbatim, but instead uses it as reference to produce well-crafted tests.

⸻

Example Prompt Usage
When you invoke this prompt for a specific file—say test_wrapped_common.md—you ask:

“Read the instructions in test_wrapped_common.md. Then produce a single Python test module that covers the code module it references, achieving >=80% coverage, and integrating best practices, property-based tests, and potential mocking, following the guidelines above.”

The response should be a valid Python file with high-quality tests, not a discussion or commentary.

⸻

End of Refined Prompt