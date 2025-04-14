# Comprehensive Guide: Using Mutation Testing with Cosmic Ray

## 1. Introduction

* **What is Mutation Testing?**
  * Mutation testing is a technique used to evaluate the quality of your software tests. It works by making small, specific changes (called "mutations") to your production code.
  * For each mutation, it runs your existing test suite.
  * **If a test fails:** The mutant is considered "killed". This is good! It means your tests detected the change.
  * **If all tests pass:** The mutant "survived". This indicates a potential weakness in your tests – they didn't notice the code change. This could mean the test coverage is insufficient, the assertions aren't strong enough, or the mutated code is "equivalent" (doesn't actually change behavior, though this is less common).
* **What is Cosmic Ray?**
  * Cosmic Ray is the mutation testing tool we use in this project. It automates the process of creating mutants, running tests, and reporting results.
* **Why Use It?**
  * To ensure our test suite is robust and catches potential bugs.
  * To identify areas where our tests might be weak or missing assertions.
  * To gain confidence in the reliability of our codebase.

## 2. Prerequisites

Before you start, make sure you have:

* **Python Environment:** A working Python 3 environment (check project requirements for specific versions).
* **Cosmic Ray Installation:** Install Cosmic Ray with:
  ```bash
  pip install cosmic-ray
  ```
* **Project Tests:** Familiarity with how to run the project's main test suite. This is typically done using pytest:
  ```bash
  pytest tests/
  ```
  Ensure this command runs successfully *before* starting mutation testing.
* **Git:** Basic understanding of Git for potentially using git-related filters later.

## 3. Core Workflow: Step-by-Step

Mutation testing with Cosmic Ray follows a standard workflow: Configure -> Initialize -> Baseline -> Execute -> Report.

* **Step 1: Configuration (`cosmic-ray.toml`)**
  * **Goal:** Tell Cosmic Ray *what* code to mutate, *how* to run the tests, and other parameters.
  * **Action:** Create a configuration file. Let's name it `cosmic-ray.toml` and place it in the project's root directory.
  * **Key Settings:**
    * `module-path`: The code to mutate. For our project, this will be `rna_predict`.
    * `test-command`: The exact command to run the test suite. Use the command identified in Prerequisites, adding `-x` to stop on the first failure (makes mutation testing faster): `"pytest tests/ -x"`.
    * `timeout`: A maximum time (in seconds) allowed for a single test run against one mutant. Start with a reasonable value like `60` or `120` and adjust if needed. This prevents infinite loops caused by mutations.
    * `excluded-modules`: A list of file patterns (globs) to *exclude* from mutation. Crucially, exclude test files themselves and potentially utility scripts or examples if they aren't the primary target. Example: `["tests/**", "docs/**"]`.
    * `distributor`: How to run the mutation jobs. Start with the simplest: `name = "local"`.
  * **Example `cosmic-ray.toml`:**
    ```toml
    # cosmic-ray.toml
    [cosmic-ray]
    module-path = "rna_predict"
    test-command = "pytest tests/ -x"
    timeout = 120.0 # 2 minutes, adjust as needed
    excluded-modules = [
        "tests/**",       # Exclude the main test directory
        "docs/**",        # Exclude documentation files
        # Add more specific exclusions if needed
    ]

    # Distributor config - start with local
    [cosmic-ray.distributor]
    name = "local"

    # Optional: Operator configuration (usually defaults are fine)
    # [cosmic-ray.operators]
    # "core/NumberReplacer" = [{}, {"offset": 10}] # Example syntax if needed

    # Optional: Filter configuration (see Advanced Topics)
    # [cosmic-ray.filters.operators-filter]
    # exclude-operators = ["core/SomeOperatorToSkip"]
    # [cosmic-ray.filters.git-filter]
    # branch = "main" # Or "master"

    # Optional: Badge configuration (see Reporting)
    # [cosmic-ray.badge]
    # label = "Mutation Score"
    # value_format = "%.1f%%"
    # [cosmic-ray.badge.thresholds]
    # 50 = "red"
    # 75 = "orange"
    # 90 = "yellow"
    # 100 = "green"
    ```
  * **Location:** Place this file at the root of the project.

* **Step 2: Initialization (`cosmic-ray init`)**
  * **Goal:** Scan the `module-path` (`rna_predict`) for all possible mutation points using the available operators and create a database of "work items".
  * **Action:** Run the following command from the project root:
    ```bash
    cosmic-ray init cosmic-ray.toml session.sqlite
    ```
  * **Output:** This creates a `session.sqlite` file. This file stores the plan (which mutations to apply where) and will later store the results.
  * **When to Re-run `init`:**
    * If you change the code in `rna_predict`.
    * If you change the `module-path`, `excluded-modules`, or operators in `cosmic-ray.toml`.
    * If you add/remove operators (e.g., by installing plugins).
    * Running `init` again *overwrites* the existing session file. If you have partial results you want to keep, don't re-run `init` unless necessary. Use `--force` to overwrite if the session file already contains results.

* **Step 3: Baselining (`cosmic-ray baseline`)**
  * **Goal:** Verify that your `test-command` (from `cosmic-ray.toml`) passes successfully on the *unmutated* code within the specified `timeout`. If the baseline fails, mutation results are meaningless.
  * **Action:** Run:
    ```bash
    cosmic-ray baseline cosmic-ray.toml
    ```
  * **Output:** Should report success. If it fails, fix your test suite or the `test-command` in the config before proceeding.

* **Step 4: Execution (`cosmic-ray exec`)**
  * **Goal:** Execute the mutation testing run. For each pending work item in `session.sqlite`:
    1. Apply the mutation to the code on disk (temporarily).
    2. Run the `test-command`.
    3. Record the outcome (killed, survived, incompetent, timeout) in `session.sqlite`.
    4. Revert the code change.
  * **Action:** Run:
    ```bash
    cosmic-ray exec cosmic-ray.toml session.sqlite
    ```
  * **Important:**
    * **Commit your code first!** While Cosmic Ray *should* always revert changes, it's safest to have a clean Git state before running `exec`.
    * This step can take a **long time**, depending on the number of mutants and the speed of your test suite.
    * You can usually **stop (`Ctrl+C`) and resume** `exec` later; it picks up from the `session.sqlite` file.
    * Use `--verbosity INFO` or `DEBUG` for more detailed progress output.

* **Step 5: Reporting (`cr-report`, `cr-html`, `cr-badge`)**
  * **Goal:** Analyze the results stored in `session.sqlite`.
  * **Actions:**
    * **Text Summary:** Get a detailed list of each mutation and its outcome.
      ```bash
      cr-report session.sqlite
      # Useful options:
      # cr-report session.sqlite --show-diff  # See code changes for each mutant
      # cr-report session.sqlite --show-output # See test output for killed/incompetent mutants
      # cr-report session.sqlite --show-pending # Include items not yet executed
      ```
    * **HTML Report:** Generate a browsable report, often including diffs.
      ```bash
      cr-html session.sqlite > cosmic-ray-report.html
      ```
      Then open `cosmic-ray-report.html` in your web browser.
    * **Survival Rate:** Get the percentage of mutants that survived. Lower is better.
      ```bash
      cr-rate session.sqlite
      ```
    * **Badge (Optional):** Generate an SVG badge showing the mutation score (kill rate). Requires configuration in `cosmic-ray.toml` (see example above).
      ```bash
      cr-badge cosmic-ray.toml badge.svg session.sqlite
      ```
  * **Interpretation:** Focus on the **surviving** mutants. Each survivor represents a gap in your test suite. Analyze the code change (`--show-diff` or HTML report) and figure out why your tests didn't fail. Then, improve your tests (add new ones, strengthen assertions) and re-run the relevant parts of the mutation testing process.

## 4. Advanced Topics: Getting the Most Out Of It

* **Filtering Mutations:**
  * Sometimes you want to skip certain mutations *before* execution.
  * **Why?** To ignore known equivalent mutants, mutations in comments/docstrings, or code explicitly marked as not needing mutation coverage. To speed up runs by focusing on specific areas.
  * **How?** Filters modify the `session.sqlite` file *after* `init` but *before* `exec`, marking items as `SKIPPED`.
  * **Available Filters (run *after* `init`):**
    * `cr-filter-pragma session.sqlite`: Skips mutations on lines containing `# pragma: no mutate`. Add this comment to your source code (`rna_predict/...`) where needed.
    * `cr-filter-operators cosmic-ray.toml session.sqlite`: Skips mutations based on operator names matching regex patterns defined in `cosmic-ray.toml` under `[cosmic-ray.filters.operators-filter]`. Useful for excluding entire categories of mutations.
    * `cr-filter-git session.sqlite`: Skips mutations on lines *not* changed relative to a specific Git branch (e.g., `main` or `master`). Configure the branch in `cosmic-ray.toml` under `[cosmic-ray.filters.git-filter]`. Great for faster feedback on pull requests.

* **Distributed Execution (Parallelism):**
  * **Why?** Mutation testing can be slow. Running jobs in parallel significantly speeds it up.
  * **How?** Use the `http` distributor.
  * **Setup:**
    1. Modify `cosmic-ray.toml`:
      * Set `[cosmic-ray.distributor]` `name = "http"`.
      * Define `[cosmic-ray.distributor.http]` `worker-urls = ["http://localhost:9876", "http://localhost:9877", ...]`. List the addresses where worker processes will listen.
    2. **Workers:** Each worker needs its *own isolated copy* of the codebase (because mutations modify files).
      * **Manual:** Clone the repository multiple times (e.g., into `worker1`, `worker2`). Start a worker in each clone directory:
        ```bash
        # In terminal 1, inside worker1 clone:
        cosmic-ray http-worker --port 9876

        # In terminal 2, inside worker2 clone:
        cosmic-ray http-worker --port 9877
        ```
      * **Helper Tool (`cr-http-workers`):** This tool automates cloning and starting local workers based on your config file. Run it from the *original* project root:
        ```bash
        # Reads worker-urls from config, clones '.', starts workers
        cr-http-workers cosmic-ray.toml .
        ```
        Run `cosmic-ray exec` in another terminal while `cr-http-workers` manages the workers. Stop `cr-http-workers` (`Ctrl+C`) when done to clean up clones.
    3. Run `cosmic-ray init` and `cosmic-ray exec` as usual from the *original* project root. `exec` will send jobs to the running workers listed in the config.

## 5. Integrating into Your Workflow

* **Local Development:**
  * Run mutation testing periodically on your feature branches.
  * Use `cr-filter-git` to focus only on changed code for faster feedback loops.
  * Analyze surviving mutants before merging to improve test quality.
* **Continuous Integration (CI/CD):**
  * **Goal:** Automatically run mutation testing on pull requests or merges.
  * **Steps in Workflow:**
    1. Checkout code.
    2. Set up Python.
    3. Install dependencies.
    4. Run `cosmic-ray init`.
    5. (Optional) Run filters (e.g., `cr-filter-pragma`).
    6. Run `cosmic-ray baseline`.
    7. Run `cosmic-ray exec`. (Consider using `cr-http-workers` if your CI runner has enough resources, or configure external workers if needed).
    8. Run `cr-report` and `cr-html`.
    9. Upload the HTML report as a build artifact.
    10. (Optional) Use `cr-rate` with the `--fail-over` option to fail the build if the survival rate is too high.
    11. (Optional) Generate and upload a badge (`cr-badge`).
  * **Challenges:** CI runs can be time-consuming. Use filters aggressively, consider parallel execution, or run mutation testing less frequently (e.g., nightly) instead of on every commit.

## 6. Best Practices and Tips

* **Start Small:** Don't try to mutate the entire codebase at once. Configure `module-path` or use exclusions/filters to target specific critical modules first.
* **Ensure Baseline Passes:** Always run `cosmic-ray baseline` first.
* **Commit Code:** Ensure your working directory is clean before running `cosmic-ray exec`.
* **Use Filters:** Employ `# pragma: no mutate` and potentially operator/git filters to reduce noise and runtime.
* **Analyze Survivors:** Don't just look at the survival rate number. Investigate *why* mutants survived and improve your tests accordingly.
* **Iterate:** Mutation testing is a cycle: test, analyze, improve tests, repeat.
* **Parallelize:** Use the `http` distributor and multiple workers (`cr-http-workers` locally or configured workers in CI) for significant speedups on larger runs.
* **Timeouts:** Set a reasonable `timeout`. Too short, and valid tests might fail (incompetent mutants); too long, and runs drag unnecessarily.

## 7. Troubleshooting Common Issues

* **Baseline Fails:** Your normal test suite is failing. Fix the tests or the `test-command` in your config.
* **High Timeout Rate:** Mutants are causing tests to hang. Increase the `timeout` value in your config, or investigate the specific mutations causing hangs (they might indicate fragile code).
* **High Survival Rate:** Your tests aren't catching the mutations. Analyze the survivors using reports (`cr-report --show-diff`, `cr-html`) and improve your tests (add assertions, cover more edge cases).
* **Configuration Errors:** Double-check the syntax and paths in your `cosmic-ray.toml`.
* **Worker Errors (HTTP Distributor):** Ensure workers are running, accessible at the configured URLs, and have the correct code version and dependencies installed in their isolated environments. Check worker logs for errors.

## 8. Checklists

* **Initial Setup Checklist:**
  * [ ] Install Cosmic Ray: `pip install cosmic-ray`
  * [ ] Verify project tests pass: `pytest tests/`
  * [ ] Create `cosmic-ray.toml` at project root.
  * [ ] Configure `module-path = "rna_predict"`.
  * [ ] Configure `test-command = "pytest tests/ -x"`.
  * [ ] Configure `timeout`.
  * [ ] Configure `excluded-modules` (especially `tests/**`).
  * [ ] Configure `distributor.name = "local"`.
* **Basic Run Checklist:**
  * [ ] Ensure code is committed or stashed.
  * [ ] Run `cosmic-ray init cosmic-ray.toml session.sqlite`.
  * [ ] (Optional) Add `# pragma: no mutate` to source files.
  * [ ] (Optional) Run `cr-filter-pragma session.sqlite`.
  * [ ] Run `cosmic-ray baseline cosmic-ray.toml`. Fix issues if it fails.
  * [ ] Run `cosmic-ray exec cosmic-ray.toml session.sqlite`.
  * [ ] Run `cr-report session.sqlite` and/or `cr-html session.sqlite > report.html`.
  * [ ] Analyze surviving mutants.
  * [ ] Improve tests based on analysis.
* **Distributed Execution Checklist (HTTP):**
  * [ ] Configure `distributor.name = "http"` in TOML.
  * [ ] Configure `distributor.http.worker-urls` in TOML.
  * [ ] Start workers:
    * Manually (clones + `cosmic-ray http-worker --port ...` in each) OR
    * Automatically (`cr-http-workers cosmic-ray.toml .`)
  * [ ] Run `init`, `baseline`, `exec` from the original project root.
  * [ ] Stop workers when done (kill manual processes or `cr-http-workers`).

## 9. Conclusion

Mutation testing is a powerful way to enhance the quality and reliability of our codebase. By systematically introducing small changes and checking if our tests detect them, we can uncover blind spots and build a more robust test suite. Start with the basic workflow, gradually incorporate filters and parallel execution, and integrate it into your development process to continuously improve test coverage and overall quality. Remember to focus on *analyzing surviving mutants* – that's where the real value lies!
