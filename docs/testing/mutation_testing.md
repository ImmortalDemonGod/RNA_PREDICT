# Mutation Testing with Mutatest

Mutation testing is a technique to evaluate the quality of your test suite by introducing small changes (mutations) to your code and checking if your tests can detect these changes.

## What is Mutation Testing?

Mutation testing works by:
1. Making small changes to your code (mutations)
2. Running your tests against the mutated code
3. If your tests fail, the mutation is "killed" (good)
4. If your tests pass, the mutation "survives" (bad)

Surviving mutations indicate areas where your tests might not be thorough enough.

## Using Mutatest in RNA_PREDICT

We use [mutatest](https://mutatest.readthedocs.io/) for mutation testing in the RNA_PREDICT project.

### Installation

```bash
pip install mutatest coverage==5.5 pytest-cov==2.12.1
```

Note: Mutatest requires specific versions of coverage and pytest-cov to work correctly.

### Configuration

The project includes a `mutatest.ini` configuration file with the following settings:

```ini
[mutatest]
exclude =
    tests/
    */__init__.py
    */scripts/*
mode = s
testcmds = pytest -n auto --cov=rna_predict tests -k 'not slow'
nocov = no
# Parallel execution is not available in this version
timeout_factor = 3.0
```

### Running Mutation Tests

We provide a script to run mutation testing:

```bash
./run_mutation_testing.sh
```

#### Script Options

- `-n, --nlocations NUMBER`: Number of locations to mutate (default: 20)
- `-m, --mode MODE`: Running mode: f, s, d, sd (default: s)
- `-o, --output FILE`: Output file for report (default: mutation_report.rst)
- `-y, --only CATEGORIES`: Only use these mutation categories (space separated)
- `-k, --skip CATEGORIES`: Skip these mutation categories (space separated)
Note: Parallel execution is not available in this version of mutatest

#### Mutation Categories

- `aa`: AugAssign (e.g., +=, -=, *=)
- `bn`: BinOp (e.g., +, -, *, /)
- `bc`: BinOpBC (e.g., &, |, ^)
- `bs`: BinOpBS (e.g., >>, <<)
- `bl`: BoolOp (e.g., and, or)
- `cp`: Compare (e.g., <, >, <=, >=, ==, !=)
- `cn`: CompareIn (e.g., in, not in)
- `cs`: CompareIs (e.g., is, is not)
- `if`: If (e.g., if statements)
- `ix`: Index (e.g., list indexing)
- `nc`: NameConstant (e.g., True, False, None)
- `su`: SliceUS (e.g., list slicing)

### Examples

Run mutation testing on 10 random locations:
```bash
./run_mutation_testing.sh -n 10
```

Run mutation testing only on comparison operators:
```bash
./run_mutation_testing.sh -y cp
```

Run mutation testing in full mode (test all possible mutations):
```bash
./run_mutation_testing.sh -m f
```

### Interpreting Results

The results are saved to `mutation_report.rst` by default. Look for:

1. **SURVIVED mutations**: These are changes that your tests didn't detect. You should improve your tests to catch these.
2. **DETECTED mutations**: These are changes that your tests successfully caught.
3. **TIMEOUT mutations**: These are changes that caused your tests to run too long.

## Best Practices

1. Start with a small number of locations (`-n 10`) to get quick feedback
2. Focus on critical modules first
3. Use the coverage filter to focus on code that's actually executed
4. Gradually increase the scope as your tests improve
5. Add tests for any surviving mutations

## Comparison with Other Tools

We previously used Cosmic Ray but switched to mutatest because:

1. Mutatest modifies the `__pycache__` files rather than source code
2. It has better isolation for parallel execution
3. It integrates well with coverage
4. It's simpler to configure and use
