Directory Structure:

└── ./
    ├── .github
    │   └── pull_request_template.md
    ├── docs
    │   ├── api_tutorial
    │   │   └── api_tutorial.rst
    │   ├── changelog.rst
    │   ├── commandline.rst
    │   ├── contributing.rst
    │   ├── index.rst
    │   ├── install.rst
    │   ├── license.rst
    │   ├── modules.rst
    │   └── mutants.rst
    ├── AUTHORS.rst
    ├── CHANGELOG.rst
    ├── CONTRIBUTING.rst
    └── README.rst



---
File: /.github/pull_request_template.md
---

<!--
Thanks for your pull-request! For efficient review please:

1. Review the [contribution guidelines](https://mutatest.readthedocs.io/en/latest/contributing.html).
2. Optionally, append your name to [Authors.rst](https://github.com/EvanKepner/mutatest/blob/master/AUTHORS.rst).
3. Include a description of the change including examples to reproduce if necessary.

Pull requests will be reviewed when the automated CI checks successfully pass.
-->

PR Checklist:

- [ ] Description of the change is included.
- [ ] Ensure all automated CI checks pass (though ask for help if needed).



---
File: /docs/api_tutorial/api_tutorial.rst
---

.. _API Tutorial:

API Tutorial
============

This is a walkthrough of using the ``mutatest`` API. These are the same
method calls used by the CLI and provide additional flexibility for
customization. The code and notebook to generate this tutorial is
located under the ``docs/api_tutorial`` folder on GitHub.

.. code:: ipython3

    # Imports used throughout the tutorial

    import ast

    from pathlib import Path

    from mutatest import run
    from mutatest import transformers
    from mutatest.api import Genome, GenomeGroup, MutationException
    from mutatest.filters import CoverageFilter, CategoryCodeFilter

Tutorial setup
--------------

The ``example/`` folder has two Python files, ``a.py`` and ``b.py``,
with a ``test_ab.py`` file that would be automatically detected by
``pytest``.

.. code:: ipython3

    # This folder and included .py files are in docs/api_tutoral/

    src_loc = Path("example")

.. code:: ipython3

    print(*[i for i in src_loc.iterdir()
            if i.is_file()], sep="\n")


.. parsed-literal::

    example/a.py
    example/test_ab.py
    example/b.py


``a.py`` holds two functions: one to add five to an input value, another
to return ``True`` if the first input value is greater than the second
input value. The add operation is represented in the AST as ``ast.Add``,
a ``BinOp`` operation type, and the greater-than operation is
represented by ``ast.Gt``, a ``CompareOp`` operation type. If the source
code is executed the expected value is to print ``10``.

.. code:: ipython3

    def open_print(fn):
        """Open a print file contents."""
        with open(fn) as f:
            print(f.read())

    # Contents of a.py example source file
    open_print(src_loc / "a.py")


.. parsed-literal::

    """Example A.
    """


    def add_five(a):
        return a + 5


    def greater_than(a, b):
        return a > b


    print(add_five(5))



``b.py`` has a single function that returns whether or not the first
input ``is`` the second input. ``is`` is represented by ``ast.Is`` and
is a ``CompareIs`` operation type. The expected value if this source
code is executed is ``True``.

.. code:: ipython3

    # Contents of b.py example source file

    open_print(src_loc / "b.py")


.. parsed-literal::

    """Example B.
    """


    def is_match(a, b):
        return a is b


    print(is_match(1, 1))



``test_ab.py`` is the test script for both ``a.py`` and ``b.py``. The
``test_add_five`` function is intentionally broken to demonstrate later
mutations. It will pass if the value is greater than 10 in the test
using 6 as an input value, and fail otherwise.

.. code:: ipython3

    # Contents of test_ab.py example test file

    open_print(src_loc / "test_ab.py")


.. parsed-literal::

    from a import add_five
    from b import is_match


    def test_add_five():
        assert add_five(6) > 10


    def test_is_match():
        assert is_match("one", "one")



Run a clean trial and generate coverage
---------------------------------------

We can use ``run`` to perform a “clean trial” of our test commands based
on the source location. This will generate a ``.coverage`` file that
will be used by the ``Genome``. A ``.coverage`` file is not required.
This run method is useful for doing clean trials before and after
mutation trials as a way to reset the ``__pycache__``.

.. code:: ipython3

    # The return value of clean_trial is the time to run
    # this is used in reporting from the CLI

    run.clean_trial(
        src_loc, test_cmds=["pytest", "--cov=example"]
    )




.. parsed-literal::

    datetime.timedelta(microseconds=411150)



.. code:: ipython3

    Path(".coverage").exists()




.. parsed-literal::

    True



Genome Basics
-------------

``Genomes`` are the basic representation of a source code file in
``mutatest``. They can be initialized by passing in the path to a
specific file, or initialized without any arguments and have the source
file added later. The basic properties include the Abstract Syntax Tree
(AST), the source file, the coverage file, and any category codes for
filtering.

.. code:: ipython3

    # Initialize with the source file location
    # By default, the ".coverage" file is set
    # for the coverage_file property

    genome = Genome(src_loc / "a.py")

.. code:: ipython3

    genome.source_file




.. parsed-literal::

    PosixPath('example/a.py')



.. code:: ipython3

    genome.coverage_file




.. parsed-literal::

    PosixPath('.coverage')



.. code:: ipython3

    # By default, no filter codes are set
    # These are categories of mutations to filter

    genome.filter_codes




.. parsed-literal::

    set()



Finding mutation targets
~~~~~~~~~~~~~~~~~~~~~~~~

The ``Genome`` has two additional properties related to finding mutation
locations: ``targets`` and ``covered_targets``. These are sets of
``LocIndex`` objects (defined in ``transformers``) that represent
locations in the AST that can be mutated. Covered targets are those that
have lines covered by the set ``coverage_file`` property.

.. code:: ipython3

    genome.targets




.. parsed-literal::

    {LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16),
     LocIndex(ast_class='Compare', lineno=10, col_offset=11, op_type=<class '_ast.Gt'>, end_lineno=10, end_col_offset=16)}



.. code:: ipython3

    genome.covered_targets




.. parsed-literal::

    {LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16)}



.. code:: ipython3

    genome.targets - genome.covered_targets




.. parsed-literal::

    {LocIndex(ast_class='Compare', lineno=10, col_offset=11, op_type=<class '_ast.Gt'>, end_lineno=10, end_col_offset=16)}



Accessing the AST
~~~~~~~~~~~~~~~~~

The ``ast`` property is the AST of the source file. You can access the
properties directly. This is used to generate the targets and covered
targets. The AST parser is defined in ``transformers`` but is
encapsulted in the ``Genome``.

.. code:: ipython3

    genome.ast




.. parsed-literal::

    <_ast.Module at 0x7f68a4014bb0>



.. code:: ipython3

    genome.ast.body




.. parsed-literal::

    [<_ast.Expr at 0x7f68a4014ca0>,
     <_ast.FunctionDef at 0x7f68a4014ac0>,
     <_ast.FunctionDef at 0x7f68a4014eb0>,
     <_ast.Expr at 0x7f68a402c040>]



.. code:: ipython3

    genome.ast.body[1].__dict__




.. parsed-literal::

    {'name': 'add_five',
     'args': <_ast.arguments at 0x7f68a4014d30>,
     'body': [<_ast.Return at 0x7f68a4014dc0>],
     'decorator_list': [],
     'returns': None,
     'type_comment': None,
     'lineno': 5,
     'col_offset': 0,
     'end_lineno': 6,
     'end_col_offset': 16}



Filtering mutation targets
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can set filters on a ``Genome`` for specific types of targets. For
example, setting ``bn`` for ``BinOp`` will filter both targets and
covered targets to only ``BinOp`` class operations.

.. code:: ipython3

    # All available categories are listed
    # in transformers.CATEGORIES

    print(*[f"Category:{k}, Code: {v}"
            for k,v in transformers.CATEGORIES.items()],
          sep="\n")


.. parsed-literal::

    Category:AugAssign, Code: aa
    Category:BinOp, Code: bn
    Category:BinOpBC, Code: bc
    Category:BinOpBS, Code: bs
    Category:BoolOp, Code: bl
    Category:Compare, Code: cp
    Category:CompareIn, Code: cn
    Category:CompareIs, Code: cs
    Category:If, Code: if
    Category:Index, Code: ix
    Category:NameConstant, Code: nc
    Category:SliceUS, Code: su


.. code:: ipython3

    # If you attempt to set an invalid code a ValueError is raised
    # and the valid codes are listed in the message

    try:
        genome.filter_codes = ("asdf",)

    except ValueError as e:
        print(e)


.. parsed-literal::

    Invalid category codes: {'asdf'}.
    Valid codes: {'AugAssign': 'aa', 'BinOp': 'bn', 'BinOpBC': 'bc', 'BinOpBS': 'bs', 'BoolOp': 'bl', 'Compare': 'cp', 'CompareIn': 'cn', 'CompareIs': 'cs', 'If': 'if', 'Index': 'ix', 'NameConstant': 'nc', 'SliceUS': 'su'}


.. code:: ipython3

    # Set the filter using an iterable of the two-letter codes

    genome.filter_codes = ("bn",)

.. code:: ipython3

    # Targets and covered targets will only show the filtered value

    genome.targets




.. parsed-literal::

    {LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16)}



.. code:: ipython3

    genome.covered_targets




.. parsed-literal::

    {LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16)}



.. code:: ipython3

    # Reset the filter_codes to an empty set
    genome.filter_codes = set()

.. code:: ipython3

    # All target classes are now listed again

    genome.targets




.. parsed-literal::

    {LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16),
     LocIndex(ast_class='Compare', lineno=10, col_offset=11, op_type=<class '_ast.Gt'>, end_lineno=10, end_col_offset=16)}



Using custom filters
~~~~~~~~~~~~~~~~~~~~

If you need more flexibility, the ``filters`` define the two classes of
filter used by ``Genome``: the ``CoverageFilter`` and the
``CategoryCodeFilter``. These are encapsultated by ``Genome`` and
``GenomeGroup`` already but can be accessed directly.

Coverage Filter
^^^^^^^^^^^^^^^

.. code:: ipython3

    cov_filter = CoverageFilter(coverage_file=Path(".coverage"))

.. code:: ipython3

    # Use the filter method to filter targets based on
    # a given source file.

    cov_filter.filter(
        genome.targets, genome.source_file
    )




.. parsed-literal::

    {LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16)}



.. code:: ipython3

    # You can invert the filtering as well

    cov_filter.filter(
        genome.targets, genome.source_file,
        invert=True
    )




.. parsed-literal::

    {LocIndex(ast_class='Compare', lineno=10, col_offset=11, op_type=<class '_ast.Gt'>, end_lineno=10, end_col_offset=16)}



Category Code Filter
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # Instantiate using a set of codes
    # or add them later

    catcode_filter = CategoryCodeFilter(codes=("bn",))

.. code:: ipython3

    # Valid codes provide all potential values

    catcode_filter.valid_codes




.. parsed-literal::

    dict_values(['aa', 'bn', 'bc', 'bs', 'bl', 'cp', 'cn', 'cs', 'if', 'ix', 'nc', 'su'])



.. code:: ipython3

    # Valid categories are also available

    catcode_filter.valid_categories




.. parsed-literal::

    {'AugAssign': 'aa',
     'BinOp': 'bn',
     'BinOpBC': 'bc',
     'BinOpBS': 'bs',
     'BoolOp': 'bl',
     'Compare': 'cp',
     'CompareIn': 'cn',
     'CompareIs': 'cs',
     'If': 'if',
     'Index': 'ix',
     'NameConstant': 'nc',
     'SliceUS': 'su'}



.. code:: ipython3

    # add more codes

    catcode_filter.add_code("aa")
    catcode_filter.codes




.. parsed-literal::

    {'aa', 'bn'}



.. code:: ipython3

    # see all validation mutations
    # based on the set codes

    catcode_filter.valid_mutations




.. parsed-literal::

    {_ast.Add,
     _ast.Div,
     _ast.FloorDiv,
     _ast.Mod,
     _ast.Mult,
     _ast.Pow,
     _ast.Sub,
     'AugAssign_Add',
     'AugAssign_Div',
     'AugAssign_Mult',
     'AugAssign_Sub'}



.. code:: ipython3

    # discard codes

    catcode_filter.discard_code("aa")
    catcode_filter.codes




.. parsed-literal::

    {'bn'}



.. code:: ipython3

    catcode_filter.valid_mutations




.. parsed-literal::

    {_ast.Add, _ast.Div, _ast.FloorDiv, _ast.Mod, _ast.Mult, _ast.Pow, _ast.Sub}



.. code:: ipython3

    # Filter a set of targets based on codes

    catcode_filter.filter(genome.targets)




.. parsed-literal::

    {LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16)}



.. code:: ipython3

    # Optionally, invert the filter

    catcode_filter.filter(
        genome.targets, invert=True
    )




.. parsed-literal::

    {LocIndex(ast_class='Compare', lineno=10, col_offset=11, op_type=<class '_ast.Gt'>, end_lineno=10, end_col_offset=16)}



Changing the source file in a Genome
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you change the source file property of the ``Genome`` all core
properties except the coverage file and filters are reset - this
includes targets, covered targets, and AST.

.. code:: ipython3

    genome.source_file = src_loc / "b.py"

.. code:: ipython3

    genome.targets




.. parsed-literal::

    {LocIndex(ast_class='CompareIs', lineno=6, col_offset=11, op_type=<class '_ast.Is'>, end_lineno=6, end_col_offset=17)}



.. code:: ipython3

    genome.covered_targets




.. parsed-literal::

    {LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16)}



Creating Mutations
------------------

Mutations are applied to specific ``LocIndex`` targets in a ``Genome``.
You must speicfy a valid operation e.g., “add” can be mutated to
“divide” or “subtract”, but not “is”. The ``Genome`` itself is not
modified, a returned ``Mutant`` object holds the information required to
create a mutated version of the ``__pycache__`` for that source file.
For this example, we’ll change ``a.py`` to use a multiplication
operation instead of an addition operation for the ``add_five``
function. The original expected result of the code was ``10`` from
``5 + 5`` if executed, with the mutation it will be ``25`` since the
mutation creates ``5 * 5``.

.. code:: ipython3

    # Set the Genome back to example a
    # filter to only the BinOp targets

    genome.source_file = src_loc / "a.py"
    genome.filter_codes = ("bn",)

    # there is only one Binop target

    mutation_target = list(genome.targets)[0]
    mutation_target




.. parsed-literal::

    LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16)



.. code:: ipython3

    # The mutate() method applies a mutation operation
    # and returns a mutant

    mutant = genome.mutate(mutation_target, ast.Mult)

.. code:: ipython3

    # applying an invalid mutation
    # raises a MutationException

    try:
        genome.mutate(mutation_target, ast.IsNot)

    except MutationException as e:
        print(e)


.. parsed-literal::

    <class '_ast.IsNot'> is not a member of mutation category bn.
    Valid mutations for bn: {<class '_ast.Mult'>, <class '_ast.Sub'>, <class '_ast.Add'>, <class '_ast.Pow'>, <class '_ast.FloorDiv'>, <class '_ast.Mod'>, <class '_ast.Div'>}.


.. code:: ipython3

    # mutants have all of the properties
    # needed to write mutated __pycache__

    mutant




.. parsed-literal::

    Mutant(mutant_code=<code object <module> at 0x7f68a4040b30, file "example/a.py", line 1>, src_file=PosixPath('example/a.py'), cfile=PosixPath('example/__pycache__/a.cpython-38.pyc'), loader=<_frozen_importlib_external.SourceFileLoader object at 0x7f689cfbd310>, source_stats={'mtime': 1571346690.5703905, 'size': 118}, mode=33188, src_idx=LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16), mutation=<class '_ast.Mult'>)



.. code:: ipython3

    # You can directly execute the mutant_code
    # This result is with the mutated target being
    # applied as Mult instead of Add in a.py

    exec(mutant.mutant_code)


.. parsed-literal::

    25


.. code:: ipython3

    # Mutants have a write_cache() method to apply
    # the change to __pycache__

    mutant.write_cache()

.. code:: ipython3

    # Alternatively, use run to do a single trial
    # and return the result

    mutant_trial_result = run.create_mutation_run_trial(
        genome, mutation_target, ast.Mult, ["pytest"], max_runtime=5
    )

.. code:: ipython3

    # In this case the mutation would survive
    # The test passes if the value is
    # greater than 10.

    mutant_trial_result.status




.. parsed-literal::

    'SURVIVED'



.. code:: ipython3

    # Using a different operation, such as Div
    # will be a detected mutation
    # since the test will fail.

    mutant_trial_result = run.create_mutation_run_trial(
        genome, mutation_target, ast.Div, ["pytest"], max_runtime=5
    )

    mutant_trial_result.status




.. parsed-literal::

    'DETECTED'



GenomeGroups
------------

The ``GenomeGroup`` is a way to interact with multiple ``Genomes``. You
can create a ``GenomeGroup`` from a folder of files, add new
``Genomes``, and access shared properties across the ``Genomes``. It is
a ``MutableMapping`` and behaves accordingly, though it only accepts
``Path`` keys and ``Genome`` values. You can use the ``GenomeGroup`` to
assign common filters, common coverage files, and to get all targets
across an entire collection of ``Genomes``.

.. code:: ipython3

    ggrp = GenomeGroup(src_loc)

.. code:: ipython3

    # key-value pairs in the GenomeGroup are
    # the path to the source file
    # and the Genome object for that file

    for k,v in ggrp.items():
        print(k, v)


.. parsed-literal::

    example/a.py <mutatest.api.Genome object at 0x7f689cfc8c10>
    example/b.py <mutatest.api.Genome object at 0x7f689cfc8f70>


.. code:: ipython3

    # targets, and covered_targets produce
    # GenomeGroupTarget objects that have
    # attributes for the source path and
    # LocIdx for the target

    for t in ggrp.targets:
        print(
            t.source_path, t.loc_idx
        )


.. parsed-literal::

    example/b.py LocIndex(ast_class='CompareIs', lineno=6, col_offset=11, op_type=<class '_ast.Is'>, end_lineno=6, end_col_offset=17)
    example/a.py LocIndex(ast_class='Compare', lineno=10, col_offset=11, op_type=<class '_ast.Gt'>, end_lineno=10, end_col_offset=16)
    example/a.py LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16)


.. code:: ipython3

    # You can set a filter or
    # coverage file for the entire set
    # of genomes

    ggrp.set_coverage = Path(".coverage")

    for t in ggrp.covered_targets:
        print(
            t.source_path, t.loc_idx
        )


.. parsed-literal::

    example/b.py LocIndex(ast_class='CompareIs', lineno=6, col_offset=11, op_type=<class '_ast.Is'>, end_lineno=6, end_col_offset=17)
    example/a.py LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>, end_lineno=6, end_col_offset=16)


.. code:: ipython3

    # Setting filter codes on all Genomes
    # in the group

    ggrp.set_filter(("cs",))
    ggrp.targets




.. parsed-literal::

    {GenomeGroupTarget(source_path=PosixPath('example/b.py'), loc_idx=LocIndex(ast_class='CompareIs', lineno=6, col_offset=11, op_type=<class '_ast.Is'>, end_lineno=6, end_col_offset=17))}



.. code:: ipython3

    for k, v in ggrp.items():
        print(k, v.filter_codes)


.. parsed-literal::

    example/a.py {'cs'}
    example/b.py {'cs'}


.. code:: ipython3

    # MutableMapping operations are
    # available as well

    ggrp.values()




.. parsed-literal::

    dict_values([<mutatest.api.Genome object at 0x7f689cfc8c10>, <mutatest.api.Genome object at 0x7f689cfc8f70>])



.. code:: ipython3

    ggrp.keys()




.. parsed-literal::

    dict_keys([PosixPath('example/a.py'), PosixPath('example/b.py')])



.. code:: ipython3

    # pop a Genome out of the Group

    genome_a = ggrp.pop(Path("example/a.py"))
    ggrp




.. parsed-literal::

    {PosixPath('example/b.py'): <mutatest.api.Genome object at 0x7f689cfc8f70>}



.. code:: ipython3

    # add a Genome to the group

    ggrp.add_genome(genome_a)
    ggrp




.. parsed-literal::

    {PosixPath('example/b.py'): <mutatest.api.Genome object at 0x7f689cfc8f70>, PosixPath('example/a.py'): <mutatest.api.Genome object at 0x7f689cfc8c10>}



.. code:: ipython3

    # the add_folder options provides
    # more flexibility e.g., to include
    # the test_ files.

    ggrp_with_tests = GenomeGroup()
    ggrp_with_tests.add_folder(
        src_loc, ignore_test_files=False
    )

    for k, v in ggrp_with_tests.items():
        print(k, v)


.. parsed-literal::

    example/a.py <mutatest.api.Genome object at 0x7f68a4044700>
    example/test_ab.py <mutatest.api.Genome object at 0x7f689cfd7340>
    example/b.py <mutatest.api.Genome object at 0x7f689cfd74f0>



---
File: /docs/changelog.rst
---

.. _Change log:

.. include:: ../CHANGELOG.rst



---
File: /docs/commandline.rst
---

.. _Command Line Controls:

Command Line Controls
=====================

Specifying source files and test commands
-----------------------------------------

If you have a Python package in a directory with an associated ``tests/`` folder
(or internal ``test_`` prefixed files, see the examples below) that are auto-detected
with ``pytest``, then you can run ``mutatest`` without any arguments.


.. code-block:: bash

    $ mutatest

It will detect the package, and run ``pytest`` by default. If you want to run with special
arguments, such as to exclude a custom marker, you can pass in the ``--testcmds`` argument
with the desired string.

Here is the command to run ``pytest`` and exclude tests marked with ``pytest.mark.slow``.

.. code-block:: bash

    $ mutatest --testcmds "pytest -m 'not slow'"

    # using shorthand arguments
    $ mutatest -t "pytest -m 'not slow'"

You can use this syntax if you want to specify a single module in your package to run and test.

.. code-block:: bash

    $ mutatest --src mypackage/run.py --testcmds "pytest tests/test_run.py"

    # using shorthand arguments
    $ mutatest -s mypackage/run.py -t "pytest tests/test_run.py"


There is an option to exclude files from the source set.
Exclude files using the ``--exclude`` argument and pointing to the file.
Multiple ``--exclude`` statements may be used to exclude multiple files. The default behavior
is that no files are excluded.

.. code-block:: bash

    $ mutatest --exclude mypackage/__init__.py --exclude mypackage/_devtools.py

    # using shorthand arguments
    $ mutatest -e mypackage/__init__.py -e mypackage/_devtools.py


These commands can all be combined in different ways to target your sample space for mutations.


Coverage filtering
-------------------

Any command combination that generates a ``.coverage`` file will use that as a restriction
mechanism for the sample space to only select mutation locations that are covered. For example,
running:

.. code-block:: bash

    $ mutatest --testcmds "pytest --cov=mypackage tests/test_run.py"

    # using shorthand arguments
    $ mutatest -t "pytest --cov=mypackage tests/test_run.py"


would generate the ``.coverage`` file based on ``tests/test_run.py``. Therefore, even though
the entire package is seen only the lines covered by ``tests/test_run.py`` will be mutated
during the trials.
If you specified a source with ``-s`` only the covered lines in that source file would become
valid targets for mutation. Excluded files with ``-e`` are still skipped.
You can override this behavior with the ``--nocov`` flag on the command line.

If you have a ``pytest.ini`` file that includes the ``--cov`` command the default behavior
of ``mutatest`` will generate the coverage file. You will see a message in the CLI output at the
beginning of the trials if coverage is ignored.

.. code-block:: bash

    # note the smaller sample based on the coverage

    $ mutatest -n 4 -t "pytest --cov=mypackage"

    ... prior output...

    ... Total sample space size: 287
    ... Selecting 4 locations from 287 potentials.
    ... Starting individual mutation trials!

    ... continued output...


    # even with coverage specified the --nocov flag is used
    # sample size is larger, and the note on ignoring is present

    $ mutatest -n 4 -t "pytest --cov=mypackage" --nocov

    ... prior output...

    ... Ignoring coverage file for sample space creation.
    ... Total sample space size: 311
    ... Selecting 4 locations from 311 potentials.
    ... Starting individual mutation trials!

    ... continued output...

.. versionadded:: 2.1.0
    Support for ``coverage`` version 4.x and 5.x.

Auto-detected package structures
--------------------------------

The following package structures would be auto-detected if you ran ``mutatest`` from the
same directory holding ``examplepkg/``. You can always point to a specific directory using
the ``--source`` argument. These are outlined in the `Pytest Test Layout`_ documentation.


Example with internal tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    .
    └── examplepkg
        ├── __init__.py
        ├── run.py
        └── test_run.py


Example with external tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    .
    ├── examplepkg
    │   ├── __init__.py
    │   └── run.py
    └── tests
        └── test_run.py



Selecting a running mode
------------------------

``mutatest`` has different running modes to make trials faster. The running modes determine
what will happen after a mutation trial. For example, you can choose to stop further mutations at a
location as soon as a survivor is detected. The different running mode choices are:

Run modes:
    - f: full mode, run all possible combinations (slowest but most thorough).
    - s: break on first SURVIVOR per mutated location e.g. if there is a single surviving mutation
      at a location move to the next location without further testing.
      This is the default mode.
    - d: break on the first DETECTION per mutated location e.g. if there is a detected mutation on
      at a location move to the next one.
    - sd: break on the first SURVIVOR or DETECTION (fastest, and least thorough).

The API for ``mutatest.controller.run_mutation_trials`` offers finer control over the run
method beyond the CLI.

A good practice when first starting is to set the mode to ``sd`` which will stop if a mutation
survives or is detected, effectively running a single mutation per candidate location. This is the
fastest running mode and can give you a sense of investigation areas quickly.

.. code-block::

    $ mutatest --mode sd

    # using shorthand arguments
    $ mutatest -m sd

Controlling randomization behavior and trial number
---------------------------------------------------

``mutatest`` uses random sampling of all source candidate locations and of potential mutations
to substitute at a location. You can set a random seed for repeatable trials using the
``--rseed`` argument. The ``--nlocations`` argument controls the size of the sample
of locations to mutate. If it exceeds the number of candidate locations then the full set of
candidate locations is used.

.. code-block::

    $ mutatest --nlocations 5 --rseed 314

    # using shorthand arguments
    $ mutatest -n 5 -r 314


Selecting categories of mutations
---------------------------------

``mutatest`` categorizes families of mutations with two-letter category codes (available in
the help output and in the mutants section below). You can use these category codes in the
``--only`` and ``--skip`` arguments to opt-in or opt-out of types of mutations
for your trials. This impacts the pool of potential locations to draw from for the sample, but the
number of mutations specified in ``--nlocations`` still determines the final sample size.
You will see the categories used in the output during the trial. Categories are space delimited
as an input list on the CLI.

.. code-block::

    # selects only the categories "aa" (AugAssign), "bn" (BinOp), and "ix" (Index) mutations
    $ mutatest --only aa bn ix

    ... prior output...

    ... Category restriction, chosen categories: ['aa', 'bn', 'ix']
    ... Setting random.seed to: None
    ... Total sample space size: 311
    ... Selecting 10 locations from 311 potentials.
    ... Starting individual mutation trials!

    ... continued output...

    # using shorthand
    $ mutatest -y aa bn ix

    # using the skip list instead, selects all categories except "aa", "bn", and "ix"
    $ mutatest --skip aa bn ix

    # with shorthand
    $ mutatest -k aa bn ix


Setting the output location
---------------------------

By default, ``mutatest`` will only create CLI output to ``stdout``.
You can set path location using the ``--output`` argument for a written RST report of the
mutation trial results.

.. code-block::

    $ mutatest --output path/to/my_custom_file.rst

    # using shorthand arguments
    $ mutatest -o path/to/my_custom_file.rst


The output report will include the arguments used to generate it along with the total runtimes.
The SURVIVORS section of the output report is the one you should pay attention to. These are the
mutations that were undetected by your test suite. The report includes file names, line numbers,
column numbers, original operation, and mutation for ease of diagnostic investigation.


Raising exceptions for survivor tolerances
------------------------------------------

By default, ``mutatest`` will only display output and not raise any final exceptions if there
are survivors in the trial results. You can set a tolerance number using the ``--exception``
or ``-x`` argument that will raise an exception if that number if met or exceeded for the
count of survivors after the trials. This argument is included for use in automated running
of ``mutatest`` e.g. as a stage in continuous integration.

When combined with the random seed and category selection you can have targeted stages for important
sections of code where you want a low count of surviving mutations enforced.

.. code-block::

    $ mutatest --exception 5

    # using shorthand arguments
    $ mutatest -x 5

The exception type is a ``SurvivingMutantException``:

.. code-block::

    ... prior output from trial...

    mutatest.cli.SurvivingMutantException: Survivor tolerance breached: 8 / 2


Controlling trial timeout behavior
----------------------------------

.. versionadded:: 1.2
    The ``--timeout_factor`` argument.

Typically mutation trials take approximately the same time as the first clean trial with some small
variance.
There are instances where a mutation could cause source code to enter an infinite loop, such
as changing a ``while`` statement using a comparison operation like ``<`` to ``>`` or ``==``.
To protect against these effects a ``--timeout_factor`` controls a multiplier of the
first clean run that will act as the timeout cap for any mutation trials.
For example, if the clean trial takes 2 seconds, and the ``--timeout_factor`` is set to 5 (the
default value), the maximum run time for a mutation trial before being stopped and logged as
a ``TIMEOUT`` is 10 seconds (2 seconds * 5).

.. code-block:: bash

    $ mutatest --timeout_factor=1.5


Note that if you set the ``--timeout_factor`` to be exactly 1 you will likely get timeout trials
by natural variance in logging success vs. failure.

.. _Parallelization:

Parallelization
---------------

.. versionadded:: 3.0.0
    Support for multiprocessing parallelization in Python 3.8.

The ``--parallel`` argument can be used if you are running with Python 3.8 to enable multiprocessing
of mutation trials. This argument has no effect if you are running Python 3.7.
Parallelism is achieved by creating parallel cache directories in a ``.mutatest_cache/`` folder
in the current working directory. Unique folders for each trial are created and the subprocess
command sets ``PYTHONPYCACHEPREFIX`` per trial. These sub-folders, and the top level
``.mutatest_cache/`` directory, are removed when the trials are complete.
Multiprocessing uses all CPUs detected by ``os.cpu_count()`` in the pool.

The parallel cache adds some IO overhead to the trial process. You will get the most benefit
from multiprocessing if you are running a longer test suite or a high number of trials.
All trials get an additional 10 seconds added to the maximum timeout calculation as a buffer
for gathering results. If you notice excessive false positive timeouts try running without
parallelization.

.. code-block:: bash

    $ mutatest --parallel


Putting it all together
-----------------------

If you want to run 5 trials, in fast ``sd`` mode, with a random seed of 345 and an output
file name of ``mutation_345.rst``, you would do the following if your directory structure
has a Python package folder and tests that are auto-discoverable and run by ``pytest``.

.. code-block:: bash

    $ mutatest -n 5 -m sd -r 345 -o mutation_345.rst


With ``coverage`` optimization if your ``pytest.ini`` file does not already specify it:

.. code-block:: bash

    $ mutatest -n 5 -m sd -r 345 -o mutation_345.rst -t "pytest --cov=mypackage"


Getting help
------------

Run ``mutatest --help`` to see command line arguments and supported operations:

.. code-block:: bash

    $ mutatest --help

    usage: Mutatest [-h] [-b [STR [STR ...]]] [-e PATH] [-m {f,s,d,sd}] [-n INT]
                    [-o PATH] [-r INT] [-s PATH] [-t STR_CMDS]
                    [-w [STR [STR ...]]] [-x INT] [--debug] [--nocov] [--parallel]
                    [--timeout_factor FLOAT > 1]

    Python mutation testing. Mutatest will manipulate local __pycache__ files.

    optional arguments:
      -h, --help            show this help message and exit
      -k [STR [STR ...]], --skip [STR [STR ...]]
                            Mutation categories to skip for trials. (default: empty list)
      -e PATH, --exclude PATH
                            Path to .py file to exclude, multiple -e entries supported. (default: None)
      -m {f,s,d,sd}, --mode {f,s,d,sd}
                            Running modes, see the choice option descriptions below. (default: s)
      -n INT, --nlocations INT
                            Number of locations in code to randomly select for mutation from possible targets. (default: 10)
      -o PATH, --output PATH
                            Output RST file location for results. (default: No output written)
      -r INT, --rseed INT   Random seed to use for sample selection.
      -s PATH, --src PATH   Source code (file or directory) for mutation testing. (default: auto-detection attempt).
      -t STR_CMDS, --testcmds STR_CMDS
                            Test command string to execute. (default: 'pytest')
      -y [STR [STR ...]], --only [STR [STR ...]]
                            Only mutation categories to use for trials. (default: empty list)
      -x INT, --exception INT
                            Count of survivors to raise Mutation Exception for system exit.
      --debug               Turn on DEBUG level logging output.
      --nocov               Ignore coverage files for optimization.
      --parallel            Run with multiprocessing (Py3.8 only).
      --timeout_factor FLOAT > 1
                            If a mutation trial running time is beyond this factor multiplied by the first
                            clean trial running time then that mutation trial is aborted and logged as a timeout.


Using a config file
-------------------

.. versionadded:: 2.2.0
    Support for ``setup.cfg`` as an optional settings file.

Arguments for ``mutatest`` can be stored in a ``mutatest.ini`` config file in the directory where
you run the command.
Use the full argument names and either spaces or newlines to separate multiple values for a given
argument.
The flag commands (``--debug`` and ``--nocov``) are given boolean flags that can be interpreted by
the Python ``ConfigParser``.
Command line arguments passed to ``mutatest`` will override the values in the ``ini`` file.
Any command line arguments that are not in the ``ini`` file will be added to the execution
parameters along with the config file values.

Alternatively, you may use ``setup.cfg`` with either a ``[mutatest]`` or ``[tool:mutatest]`` entry.
The ``mutatest.ini`` file will be used first if it is present, skipping ``setup.cfg``.
``setup.cfg`` will honor the ``[mutatest]`` and ``[tool:mutatest]`` in that order.
Entries are not combined if both are present.


Example config file
~~~~~~~~~~~~~~~~~~~

The contents of an example ``mutatest.ini`` or entry in ``setup.cfg``:

.. code-block:: ini

   [mutatest]

   skip = nc su ix
   exclude =
       mutatest/__init__.py
       mutatest/_devtools.py
   mode = sd
   rseed = 567
   testcmds = pytest -m 'not slow'
   debug = no
   nocov = no


.. target-notes::
.. _Pytest Test Layout: https://docs.pytest.org/en/latest/goodpractices.html#choosing-a-test-layout-import-rules
.. _Python AST grammar: https://docs.python.org/3/library/ast.html#abstract-grammar



---
File: /docs/contributing.rst
---

.. _contributing:

.. include:: ../CONTRIBUTING.rst



---
File: /docs/index.rst
---

Mutatest: Python mutation testing
=================================

|  |py-versions| |license| |ci-azure| |ci-travis| |docs| |coverage| |black|
|  |pypi-version| |pypi-status| |pypi-format| |pypi-downloads|
|  |conda-version| |conda-recipe| |conda-platform| |conda-downloads|


Are you confident in your tests? Try out ``mutatest`` and see if your tests will detect small
modifications (mutations) in the code. Surviving mutations represent subtle changes that are
undetectable by your tests. These mutants are potential modifications in source code that continuous
integration checks would miss.

Features:
---------

    - Simple command line tool with `multiple configuration options <https://mutatest.readthedocs.io/en/latest/commandline.html>`_.
    - Built on Python's Abstract Syntax Tree (AST) grammar to ensure `mutants are valid <https://mutatest.readthedocs.io/en/latest/mutants.html>`_.
    - `No source code modification <https://mutatest.readthedocs.io/en/latest/install.html#mutation-trial-process>`_,
      only the ``__pycache__`` is changed.
    - Uses ``coverage`` to create `only meaningful mutants <https://mutatest.readthedocs.io/en/latest/commandline.html#coverage-filtering>`_.
    - Built for efficiency with `multiple running modes <https://mutatest.readthedocs.io/en/latest/commandline.html#selecting-a-running-mode>`_
      and `random sampling of mutation targets <https://mutatest.readthedocs.io/en/latest/commandline.html#controlling-randomization-behavior-and-trial-number>`_.
    - Capable of running `parallel mutation trials <https://mutatest.readthedocs.io/en/latest/commandline.html#parallelization>`_
      with multiprocessing on Python 3.8.
    - Flexible enough to run on a `whole package <https://mutatest.readthedocs.io/en/latest/commandline.html#auto-detected-package-structures>`_
      or a `single file <https://mutatest.readthedocs.io/en/latest/commandline.html#specifying-source-files-and-test-commands>`_.
    - Includes an `API for custom mutation controls <https://mutatest.readthedocs.io/en/latest/modules.html>`_.
    - Tested on Linux, Windows, and MacOS with `Azure pipelines <https://dev.azure.com/evankepner/mutatest/_build/latest?definitionId=1&branchName=master>`_.
    - Full strict static type annotations throughout the source code and the API.


Quick Start
-----------

``mutatest`` requires Python 3.7 or 3.8.

Install from `PyPI <https://pypi.org/project/mutatest/>`_:

.. code-block:: bash

    $ pip install mutatest

Install from `conda-forge <https://anaconda.org/conda-forge/mutatest>`_:

.. code-block:: bash

    $ conda install -c conda-forge mutatest


Alternatively, clone the repo from `GitHub <https://github.com/EvanKepner/mutatest>`_ and install
from the source code:

.. code-block:: bash

    $ cd mutatest
    $ pip install .


``mutatest`` is designed to work when your test files are separated from your source directory
and are prefixed with ``test_``.
See `Pytest Test Layout <https://docs.pytest.org/en/latest/goodpractices.html#choosing-a-test-layout-import-rules>`_
for more details.


``mutatest`` is a diagnostic command line tool for your test coverage assessment.
If you have a Python package in with an associated ``tests/`` folder, or internal ``test_`` prefixed files,
that are auto-detected with ``pytest``, then you can run ``mutatest`` without any arguments.


.. code-block:: bash

    $ mutatest

See more examples with additional configuration options in :ref:`Command Line Controls`.


Help
~~~~

Run ``mutatest --help`` to see command line arguments and supported operations:

.. code-block:: bash

    $ mutatest --help

    usage: Mutatest [-h] [-b [STR [STR ...]]] [-e PATH] [-m {f,s,d,sd}] [-n INT]
                    [-o PATH] [-r INT] [-s PATH] [-t STR_CMDS]
                    [-w [STR [STR ...]]] [-x INT] [--debug] [--nocov] [--parallel]
                    [--timeout_factor FLOAT > 1]

    Python mutation testing. Mutatest will manipulate local __pycache__ files.

    optional arguments:
      -h, --help            show this help message and exit
      -k [STR [STR ...]], --skip [STR [STR ...]]
                            Mutation categories to skip for trials. (default: empty list)
      -e PATH, --exclude PATH
                            Path to .py file to exclude, multiple -e entries supported. (default: None)
      -m {f,s,d,sd}, --mode {f,s,d,sd}
                            Running modes, see the choice option descriptions below. (default: s)
      -n INT, --nlocations INT
                            Number of locations in code to randomly select for mutation from possible targets. (default: 10)
      -o PATH, --output PATH
                            Output RST file location for results. (default: No output written)
      -r INT, --rseed INT   Random seed to use for sample selection.
      -s PATH, --src PATH   Source code (file or directory) for mutation testing. (default: auto-detection attempt).
      -t STR_CMDS, --testcmds STR_CMDS
                            Test command string to execute. (default: 'pytest')
      -y [STR [STR ...]], --only [STR [STR ...]]
                            Only mutation categories to use for trials. (default: empty list)
      -x INT, --exception INT
                            Count of survivors to raise Mutation Exception for system exit.
      --debug               Turn on DEBUG level logging output.
      --nocov               Ignore coverage files for optimization.
      --parallel            Run with multiprocessing (Py3.8 only).
      --timeout_factor FLOAT > 1
                            If a mutation trial running time is beyond this factor multiplied by the first
                            clean trial running time then that mutation trial is aborted and logged as a timeout.


Example Output
~~~~~~~~~~~~~~

This is an output example running mutation trials against the :ref:`API Tutorial` example folder.

.. code-block:: bash

    $ mutatest -s example/ -t "pytest" -r 314

    Running clean trial
    2 mutation targets found in example/a.py AST.
    1 mutation targets found in example/b.py AST.
    Setting random.seed to: 314
    Total sample space size: 2
    10 exceeds sample space, using full sample: 2.

    Starting individual mutation trials!
    Current target location: a.py, LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>)
    Detected mutation at example/a.py: (6, 11)
    Detected mutation at example/a.py: (6, 11)
    Surviving mutation at example/a.py: (6, 11)
    Break on survival: stopping further mutations at location.

    Current target location: b.py, LocIndex(ast_class='CompareIs', lineno=6, col_offset=11, op_type=<class '_ast.Is'>)
    Detected mutation at example/b.py: (6, 11)
    Running clean trial

    Mutatest diagnostic summary
    ===========================
     - Source location: /home/user/Github/mutatest/docs/api_tutorial/example
     - Test commands: ['pytest']
     - Mode: s
     - Excluded files: []
     - N locations input: 10
     - Random seed: 314

    Random sample details
    ---------------------
     - Total locations mutated: 2
     - Total locations identified: 2
     - Location sample coverage: 100.00 %


    Running time details
    --------------------
     - Clean trial 1 run time: 0:00:00.348999
     - Clean trial 2 run time: 0:00:00.350213
     - Mutation trials total run time: 0:00:01.389095

    Trial Summary Report:

    Overall mutation trial summary
    ==============================
     - DETECTED: 3
     - SURVIVED: 1
     - TOTAL RUNS: 4
     - RUN DATETIME: 2019-10-17 16:57:08.645355

    Detected mutations:

    DETECTED
    --------
     - example/a.py: (l: 6, c: 11) - mutation from <class '_ast.Add'> to <class '_ast.Sub'>
     - example/a.py: (l: 6, c: 11) - mutation from <class '_ast.Add'> to <class '_ast.Mod'>
     - example/b.py: (l: 6, c: 11) - mutation from <class '_ast.Is'> to <class '_ast.IsNot'>

    Surviving mutations:

    SURVIVED
    --------
     - example/a.py: (l: 6, c: 11) - mutation from <class '_ast.Add'> to <class '_ast.Mult'>


Contents
========


.. toctree::
   :maxdepth: 4

   install
   commandline
   mutants
   api_tutorial/api_tutorial
   modules
   license
   changelog
   contributing
   GitHub <https://github.com/EvanKepner/mutatest>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |py-versions| image:: https://img.shields.io/pypi/pyversions/mutatest?color=green
    :target: https://pypi.org/project/mutatest/
    :alt: Python versions
.. |license| image:: https://img.shields.io/pypi/l/mutatest.svg
    :target: https://pypi.org/project/mutatest/
    :alt: License
.. |pypi-version| image:: https://badge.fury.io/py/mutatest.svg
    :target: https://pypi.org/project/mutatest/
    :alt: PyPI version
.. |pypi-status| image:: https://img.shields.io/pypi/status/mutatest.svg
    :target: https://pypi.org/project/mutatest/
    :alt: PyPI status
.. |pypi-format| image:: https://img.shields.io/pypi/format/mutatest.svg
    :target: https://pypi.org/project/mutatest/
    :alt: PyPI Format
.. |pypi-downloads| image:: https://pepy.tech/badge/mutatest
    :target: https://pepy.tech/project/mutatest
    :alt: PyPI Downloads
.. |ci-travis| image:: https://travis-ci.org/EvanKepner/mutatest.svg?branch=master
    :target: https://travis-ci.org/EvanKepner/mutatest
    :alt: TravisCI
.. |ci-azure| image:: https://dev.azure.com/evankepner/mutatest/_apis/build/status/EvanKepner.mutatest?branchName=master
    :target: https://dev.azure.com/evankepner/mutatest/_build/latest?definitionId=1&branchName=master
    :alt: Azure Pipelines
.. |docs| image:: https://readthedocs.org/projects/mutatest/badge/?version=latest
    :target: https://mutatest.readthedocs.io/en/latest/?badge=latest
    :alt: RTD status
.. |coverage| image:: https://codecov.io/gh/EvanKepner/mutatest/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/EvanKepner/mutatest
    :alt: CodeCov
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black
.. |conda-recipe| image:: https://img.shields.io/badge/recipe-mutatest-green.svg
    :target: https://anaconda.org/conda-forge/mutatest
    :alt: Conda recipe
.. |conda-version| image:: https://img.shields.io/conda/vn/conda-forge/mutatest.svg
    :target: https://anaconda.org/conda-forge/mutatest
    :alt: Conda version
.. |conda-platform| image:: https://img.shields.io/conda/pn/conda-forge/mutatest.svg
    :target: https://anaconda.org/conda-forge/mutatest
    :alt: Conda platforms
.. |conda-azure| image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/mutatest-feedstock?branchName=master
    :target: https://anaconda.org/conda-forge/mutatest
    :alt: Conda azure status
.. |conda-downloads| image:: https://img.shields.io/conda/dn/conda-forge/mutatest.svg
    :target: https://anaconda.org/conda-forge/mutatest
    :alt: Conda downloads



---
File: /docs/install.rst
---

.. _Installation:

Installation
============

``mutatest`` requires Python 3.7 or Python 3.8.


Install from `PyPI <https://pypi.org/project/mutatest/>`_:

.. code-block:: bash

    $ pip install mutatest

Install from `conda-forge <https://anaconda.org/conda-forge/mutatest>`_:

.. code-block:: bash

    $ conda install -c conda-forge mutatest


Alternatively, clone this repo and install locally:

.. code-block:: bash

    $ cd mutatest
    $ pip install .


``mutatest`` is designed to work when your test files are separated from your source directory
and are prefixed with ``test_``. See `Pytest Test Layout`_ for more details.


.. _Mutation Trial Process:

Mutation Trial Process
======================

``mutatest`` is designed to be a diagnostic command line tool for your test coverage assessment.

The mutation trial process follows these steps when ``mutatest`` is run from the CLI:

1. Scan for your existing Python package, or use the input source location.
2. Create an abstract syntax tree (AST) from the source files.
3. Identify locations in the code that may be mutated (line and column). If you are running with
   ``coverage`` the sample is restricted only to lines that are marked as covered in the
   ``.coverage`` file.
4. Take a random sample of the identified locations.
5. Apply a mutation at the location by modifying a copy of the AST and writing a new cache file
   to the appropriate ``__pycache__`` location with the source file statistics.
6. Run the test suite. This will use the mutated ``__pycache__`` file since the source statistics
   are the same for modification time.
7. See if the test suite detected the mutation by a failed test.
8. Remove the modified ``__pycache__`` file.
9. Repeat steps 5-9 for the remaining selected locations to mutate.
10. Write an output report of the various mutation results.

A "clean trial" of your tests is run before any mutations are applied. This same "clean trial" is
run at the end of the mutation testing. This ensures that your original test suite passes before
attempting to detect surviving mutations and that the ``__pycache__`` has been appropriately
reset when the mutation trials are finished.

.. _Motivation:

Motivation and FAQs
===================

Mutation Testing Overview
-------------------------

Mutation testing is designed to assess the quality of other testing; typically, unit tests.
The idea is that unit tests should fail given a specific mutation in a tested function.
For example, if a new contributor were to submit a pull request for an important numerical library
and accidentally typo a ``>`` to be ``>=`` in an existing tested function, the maintainer should
expect that the change is detected through unit test failure.
Mutation testing is a way to ensure this assumption is valid.
Essentially, mutation testing is a test of the alarm system created by the unit tests.


Why random sampling instead of all possible mutants?
----------------------------------------------------

By nature, mutation testing can be slow.
You have to make a small modification in your source code and then see if your test suite fails.
For fast tests and smaller projects running every possible mutation may be feasible.
For larger projects, this could be prohibitively expensive in time.
Random sampling of the target locations, as well as of the mutations to apply, takes advantage
of the "alarm testing" nature of mutation testing.
You do not need to exhaustively test every mutation to have a good understanding of whether or not
your test suite is generally sufficient to detect these changes, and it provides a sense of
the types of mutations that could slip past your unit tests.
Using the source and test commands targeting, as well as the category filters, you can create specific
mutation trials for important components of your code.
Setting a `random seed <https://mutatest.readthedocs.io/en/latest/commandline.html#controlling-randomization-behavior-and-trial-number>`_
on the command line ensures reproducibility for the same set of arguments.

Why modify the pycache?
-----------------------

In short, protection of source code.
A goal of ``mutatest`` is to avoid source code modification so that mutations are not accidentally
committed to version control.
Writing the mutations from memory to the ``__pycache__`` is a safety mechanism to ensure that the
worst-case scenario of a killed process in a trial is to clear you cache.


Can I use mutatest in CICD?
---------------------------

Yes, though because of the slow nature of running your test suite multiple times it is not something
you would run across your entire codebase on every commit.
``Mutatest`` includes an option to `raise survivor exceptions <https://mutatest.readthedocs.io/en/latest/commandline.html#raising-exceptions-for-survivor-tolerances>`_
based on a tolerance level e.g., you may tolerate up to 2 surviving mutants (you set the threshold)
out of 20 with specific pieces of your source code.
``Mutatest`` is most useful as a diagnostic tool to determine weak spots in your overall test structure.


Are there differences in running with Python 3.7 vs. Python 3.8?
----------------------------------------------------------------

.. versionadded:: 2.0.0
    Support for Python 3.8
.. versionadded:: 3.0.0
    Multiprocessing parallelization in Python 3.8

Yes, though they do not impact the command line interface experience.
In Python 3.8, the ``NamedConstant`` node type was deprecated in favor of ``Constant``, and new
location attributes were added to individual nodes: ``end_lineno`` and ``end_col_offset``.
These changes are accounted for in the ``transformers`` module.
Running with Python 3.7 the ``LocIndex.end_lineno`` and ``LocIndex.end_col_offset`` will always
be set to ``None``, and in Python 3.8 these values are populated based on the AST.
Additional information is on `Python 3.8 What's New Improved Modules`_.

Python 3.8 also supports a parallel pycache directory. This is used to enable multiprocessing of
mutation trials with the ``--parallel`` argument. Parallelization is not supported on Python 3.7.


Known limitations
-----------------

Since ``mutatest`` operates on the local ``__pycache__`` it is a serial execution process.
This means it will take as long as running your test suite in series for the
number of operations. It's designed as a diagnostic tool, and you should try to find the combination
of test commands, source specifiers, and exclusions that generate meaningful diagnostics.
For example, if you have 600 tests, running ``mutatest`` over the entire test suite may take
some time. A better strategy would be:

1. Select a subset of your tests and run ``pytest`` with ``coverage`` to see the
   covered percentage per source file.
2. Run ``mutatest`` with the same ``pytest`` command passed in with ``-t`` and generating
   a coverage file. Use ``-s`` to pick the source file of interest to restrict the sample space,
   or use ``-e`` to exclude files if you want to target multiple files.


If you kill the ``mutatest`` process before the trials complete you may end up
with partially mutated ``__pycache__`` files. If this happens the best fix is to remove the
``__pycache__`` directories and let them rebuild automatically the next time your package is
imported (for instance, by re-running your test suite).

The mutation status is based on the return code of the test suite e.g. 0 for success, 1 for failure.
``mutatest`` can theoretically be run with any test suite that you pass with the
``--testcmds`` argument; however, only ``pytest`` has been tested to date. The
``mutatest.run.MutantTrialResult`` contains the definitions for translating
return codes into mutation trial statuses.

.. target-notes::
.. _Pytest Test Layout: https://docs.pytest.org/en/latest/goodpractices.html#choosing-a-test-layout-import-rules
.. _Python 3.8 What's New Improved Modules: https://docs.python.org/3/whatsnew/3.8.html#ast



---
File: /docs/license.rst
---

.. _license:

License
=======

Distributed under the terms of the `MIT`_ license, ``mutatest`` is free and open source software.

.. literalinclude:: ../LICENSE

.. target-notes::
.. _`MIT`: https://github.com/EvanKepner/mutatest/blob/master/LICENSE



---
File: /docs/modules.rst
---

.. _API Reference:

API Reference
=============

.. automodule:: mutatest.api
   :members:

.. automodule:: mutatest.cache
   :members:

.. automodule:: mutatest.cli
   :members:

.. automodule:: mutatest.filters
   :members:

.. automodule:: mutatest.report
   :members:

.. automodule:: mutatest.run
   :members:

.. automodule:: mutatest.transformers
   :members:



---
File: /docs/mutants.rst
---

.. _Mutations:

Mutations
=========

``mutatest`` supports the following mutation operations based on the `Python AST grammar`_:

Supported operations:
    - ``AugAssign`` mutations e.g. ``+= -= *= /=``.
    - ``BinOp`` mutations e.g. ``+ - / *``.
    - ``BinOp Bitwise Comparison`` mutations e.g. ``x&y x|y x^y``.
    - ``BinOp Bitwise Shift`` mutations e.g. ``<< >>``.
    - ``BoolOp`` mutations e.g. ``and or``.
    - ``Compare`` mutations e.g. ``== >= < <= !=``.
    - ``Compare In`` mutations e.g. ``in, not in``.
    - ``Compare Is`` mutations e.g. ``is, is not``.
    - ``If`` mutations e.g. ``If x > y`` becomes ``If True`` or ``If False``.
    - ``Index`` mutations e.g. ``i[0]`` becomes ``i[1]`` and ``i[-1]``.
    - ``NameConstant`` mutations e.g. ``True``, ``False``, and ``None``.
    - ``Slice`` mutations e.g. changing ``x[:2]`` to ``x[2:]``.

These are the current operations that are mutated as compatible sets.
The two-letter category code for white/black-list selection is beside the name in double quotes.


AugAssign - "aa"
----------------

Augmented assignment e.g. ``+= -= /= *=``.

Members:
    - ``AugAssign_Add``
    - ``AugAssign_Div``
    - ``AugAssign_Mult``
    - ``AugAssign_Sub``


Example:

.. code-block:: python

    # source code
    x += y

    # mutations
    x -= y  # AugAssign_Sub
    x *= y  # AugAssign_Mult
    x /= y  # AugAssign_Div


BinOp - "bn"
------------

Binary operations e.g. add, subtract, divide, etc.

Members:
    - ``ast.Add``
    - ``ast.Div``
    - ``ast.FloorDiv``
    - ``ast.Mod``
    - ``ast.Mult``
    - ``ast.Pow``
    - ``ast.Sub``


Example:

.. code-block:: python

    # source code
    x = a + b

    # mutations
    x = a / b  # ast.Div
    x = a - b  # ast.Sub


BinOp Bit Comparison - "bc"
---------------------------

Bitwise comparison operations e.g. ``x & y, x | y, x ^ y``.

Members:
    - ``ast.BitAnd``
    - ``ast.BitOr``
    - ``ast.BitXor``


Example:

.. code-block:: python

    # source code
    x = a & y

    # mutations
    x = a | y  # ast.BitOr
    x = a ^ y  # ast.BitXor


BinOp Bit Shifts - "bs"
-----------------------

Bitwise shift operations e.g. ``<< >>``.

Members:
    - ``ast.LShift``
    - ``ast.RShift``

Example:

.. code-block:: python

    # source code
    x >> y

    # mutation
    x << y

BoolOp - "bl"
-------------

Boolean operations e.g. ``and or``.

Members:
    - ``ast.And``
    - ``ast.Or``


Example:

.. code-block:: python

    # source code
    if x and y:

    # mutation
    if x or y:


Compare - "cp"
--------------

Comparison operations e.g. ``== >= <= > <``.

Members:
    - ``ast.Eq``
    - ``ast.Gt``
    - ``ast.GtE``
    - ``ast.Lt``
    - ``ast.LtE``
    - ``ast.NotEq``

Example:

.. code-block:: python

    # source code
    x >= y

    # mutations
    x < y  # ast.Lt
    x > y  # ast.Gt
    x != y  # ast.NotEq


Compare In - "cn"
-----------------

Compare membership e.g. ``in, not in``.

Members:
    - ``ast.In``
    - ``ast.NotIn``


Example:

.. code-block:: python

    # source code
    x in [1, 2, 3, 4]

    # mutation
    x not in [1, 2, 3, 4]


Compare Is - "cs"
-----------------

Comapre identity e.g. ``is, is not``.

Members:
    - ``ast.Is``
    - ``ast.IsNot``

Example:

.. code-block:: python

    # source code
    x is None

    # mutation
    x is not None


If - "if"
---------

If mutations change ``if`` statements to always be ``True`` or ``False``. The original
statement is represented by the class ``If_Statement`` in reporting.

Members:
    - ``If_False``
    - ``If_Statement``
    - ``If_True``


Example:

.. code-block:: python

    # source code
    if a > b:   # If_Statement
        ...

    # Mutations
    if True:   # If_True
        ...

    if False:  # If_False
        ...


Index - "ix"
------------

Index values for iterables e.g. ``i[-1], i[0], i[0][1]``. It is worth noting that this is a
unique mutation form in that any index value that is positive will be marked as ``Index_NumPos``
and the same relative behavior will happen for negative index values to ``Index_NumNeg``. During
the mutation process there are three possible outcomes: the index is set to 0, -1 or 1.
The alternate values are chosen as potential mutations e.g. if the original operation is classified
as ``Index_NumPos`` such as ``x[10]`` then valid mutations are to ``x[0]`` or
``x[-1]``.

Members:
    - ``Index_NumNeg``
    - ``Index_NumPos``
    - ``Index_NumZero``


Example:

.. code-block:: python

    # source code
    x = [a[10], a[-4], a[0]]

    # mutations
    x = [a[-1], a[-4], a[0]]  # a[10] mutated to Index_NumNeg
    x = [a[10], a[0], a[0]]  # a[-4] mutated to Index_NumZero
    x = [a[10], a[1], a[0]]  # a[-4] mutated to Index_NumPos
    x = [a[10], a[-4], a[1]]  # a[0] mutated to Index_NumPos


NameConstant - "nc"
-------------------

Named constant mutations e.g. ``True, False, None``.

Members:
    - ``False``
    - ``None``
    - ``True``


Example:

.. code-block:: python

    # source code
    x = True

    # mutations
    x = False
    X = None


Slices - "su"
-------------

Slice mutations to swap lower/upper values, or change range e.g. ``x[2:] to x[:2]``.
This is a unique mutation. If the upper or lower bound is set to
``None`` then the bound values are swapped. This is represented by the operations of
``Slice_UnboundedUpper`` for swap None to the "upper" value  from "lower". The category code
for this type of mutation is "su".

Members:
    - ``Slice_Unbounded``
    - ``Slice_UnboundedLower``
    - ``Slice_UnboundedUpper``


Example:

.. code-block:: python

    # source code
    w = a[:2]
    x = a[4:]

    # mutation
    w = a[2:]  # Slice_UnboundedUpper, upper is now unbounded and lower has a value
    x = a[4:]

    # mutation
    w = a[:2]
    x = a[:4]  # Slice_UnboundedLower, lower is now unbounded and upper has a value

    # mutation
    w = a[:2]
    x = a[:]  # Slice_Unbounded, both upper and lower are unbounded


.. target-notes::
.. _Python AST grammar: https://docs.python.org/3/library/ast.html#abstract-grammar



---
File: /AUTHORS.rst
---

Authors
=======

``mutatest`` is written and maintained by Evan Kepner.

See the `Contributing Guidelines <https://mutatest.readthedocs.io/en/latest/contributing.html>`_ if you
are interested in submitting code in the form of pull requests.
Contributors are listed here, and can be seen on the
`GitHub contribution graph <https://github.com/EvanKepner/mutatest/graphs/contributors>`_.

Contributors
------------

* David Li-Bland
* Alireza Aghamohammadi



---
File: /CHANGELOG.rst
---

Changelog
=========

Stable Releases
---------------

3.1.0
~~~~~

  - Maintenance patches and API changes to skip/only category selection.

3.0.2
~~~~~

    - `Maintenance patch #27 <https://github.com/EvanKepner/mutatest/pull/27>`_ updating source
      code conditional logic in the CLI argument parsing.
    - Minor fixes for the most updated CI checks.


3.0.1
~~~~~

    - `Bug fix #24 <https://github.com/EvanKepner/mutatest/issues/24>`_ where the bit-shift
      operators where not being applied during mutation trials and raised ``KeyError``.
    - A new ``test_all_op_types.py`` ensures all mutation substitutions work as intended.


3.0.0
~~~~~

    - ``Mutatest`` has reached a level of maturity to warrant a stable release.
      With the addition of the multiprocessing capabilities, support for ``coverage`` versions
      4.x and 5.x, support for Python 3.7 and 3.8, being installable through ``pip`` or
      ``conda``, and with Azure Pipelines CI for platform tests, the tool and API are
      unlikely to change in a major way without moving to ``4.0.0``.

    New in this release:

    - Multiprocessing support on Python 3.8!
        - The new ``--parallel`` command argument will instruct ``mutatest`` to use
          multiprocessing for mutation trials. See the documentation for complete details.

    - Bug fix in ``mutatest.cache.create_cache_dirs()`` where the cache directory did not
      include "parents" in case of packages with nested directories without existing pycache.
    - Removal of the ``sr`` subcategory of slice mutations (``Slice_RC`` for range changes).
      These were rare, and supporting both Python 3.7 and 3.8 required excessive complexity.
      The ``su`` category remains valid as the primary slice mutation set.


Beta Releases
-------------

2.2.0
~~~~~

    - Added support for specifying settings in ``setup.cfg`` using either ``[mutatest]`` or
      ``[tool:mutatest]`` sections in addition to the ``mutatest.ini`` file.

2.1.3
~~~~~

    - Addressing test issues on Windows platform in the coverage tests by adding a
      ``resolve_source`` flag to the ``CoverageFilter.filter`` method.

2.1.2
~~~~~

    - Moved the ``tests`` directory to be within the package of ``mutatest``.
      This enabled the installation to be tested with ``pytest --pyargs mutatest`` as well
      as ``pytest`` from local source files.
      Test dependencies are still installed with ``pip install .[tests]``.

2.1.1
~~~~~

    - Includes specific test environments for ``coverage`` versions 4 and 5 with appropriate mocked
      ``.coverage`` data outputs (JSON or SQL based on version).
    - A new ``tox`` test environment called ``cov4`` is added, with a new ``pytest`` marker
      ``pytest.mark.coverage`` for test selection.

2.1.0
~~~~~

    - ``Coverage`` version 5.0 has moved to a SQLite database instead of a flat file. To support
      both 4x and 5x versions of ``Coverage`` the ``filters`` source code has been updated.
      The test suite includes mocked coverage data parsing tests of 4x only for now.

2.0.1
~~~~~

    - Explicit including of ``typing-extensions`` in ``setup.py`` requirements to fix breaking
      documentation builds on Python version 3.7 vs. 3.8.

2.0.0
~~~~~

    - Python 3.8 support! There are breaking changes with the ``LocIndex`` and other components
      of the ``transformers`` from prior versions of ``mutatest``. Python 3.8 introduces a new
      AST structure - including additional node attributes ``end_lineno`` and ``end_col_offset``
      that have to be accounted for. ``transformers.MutateAST`` is now build from a base class
      and a mixin class depending on the Python version (3.7 vs. 3.8) for the appropriate AST
      treatment. There are no changes in the CLI usage.


1.2.1
~~~~~

    - Bugfix to ensure ``exclude`` path processing in ``GenomeGroup.add_folder`` always uses full
      resolved paths for files.

1.2.0
~~~~~

    - `Feature #18 <https://github.com/EvanKepner/mutatest/pull/18>`_: Allow mutation trials to time out.
      There are cases where a mutation could cause an infinite loop, such as changing the comparator in
      a ``while`` statement e.g., ``while x < 5`` becomes ``while x >= 5``. A new ``--timeout_factor``
      argument is added to set a cap on the maximum trial time as a multiplier of the clean-trial run.
    - Bugfix on using ``exclude`` where files were logged but still becoming part of the sample.

1.1.1
~~~~~

    - `Bug Fix #15 <https://github.com/EvanKepner/mutatest/pull/15>`_: Fix ``LocIndex.ast_class`` setting for ``Index`` node mutations.


1.1.0
~~~~~

    - Add support for a ``mutatest.ini`` configuration file for command line arguments.


1.0.1
~~~~~

    - Documentation updates, including the API tutorial.
    - Fix on latest ``mypy`` errors related to ``strict`` processing of ``run`` and ``cache``.


1.0.0
~~~~~

    - Moving from the alpha to the beta version with an API design. The alpha releases were focused
      on defining the functionality of the CLI. In the beta version, the CLI remains unchanged; however,
      a full internal design has been applied to create a coherent API. The ``controller``, ``optimizers``,
      and ``maker`` modules have been fully replaced by ``run``, ``api``, and ``filters``. See
      the new full API documentation for details on using these modules outside of the CLI.
    - Additionally, ``pytest`` was removed from the installation requirements since it is assumed
      for the default running modes but not required for the API or installation.


Alpha Releases
--------------

0.9.2
~~~~~

    - Added ``--exception`` and ``-x`` as a survivor tolerance to raise an exception
      after the trial completes if the count of surviving mutants is greater than or equal to the
      specified value.

0.9.1
~~~~~

    - Added ``--only`` and ``--skip`` with category codes for mutation families.
    - Provides CLI selection of mutation types to be used during the trials.


0.9.0
~~~~~

    - Added new ``If`` mutation:
        1. Original statements are represented by ``If_Statement`` and mutated to be either
           ``If_True`` where the statement always passes, or ``If_False`` where the statement
           is never passed.


0.8.0
~~~~~

    - Breaking changes to the CLI arguments and new defaults:
        1. Output files are now optional, the default behavior has changed from always writing an RST
           file using the ``-o`` option on the command line.
        2. Exclusions are still marked as ``-e``; however, now multiple ``-e`` arguments are
           supported and arguments must point to a Python file. The argument used to be:
           ``mutatest -e "__init__.py _devtools.py"`` and now it is
           ``mutatest -e src/__init__.py -e src/_devtools.py``. There are no longer default
           exclusions applied.

    - Improved CLI reporting, including selected test counts and line/col locations
      for trial results while processing.


0.7.1
~~~~~

    - Internal changes to ``Slice`` mutations for clearer categorization and report output.
    - Includes clearing names to ``Slice_Swap`` and ``Slice_RangeChange`` for categories.
    - Updates operation names to ``Slice_Unbounded...`` with "lower" or "upper".

0.7.0
~~~~~

    - Added new slice mutations:
        1. ``Slice_SwapNoneUL`` and ``Slice_SwapNoneLU`` for swapping the upper and lower
           bound values when only one is specified e.g. ``x[1:]`` to ``x[:1]``.
        2. ``Slice_UPosToZero`` and ``Slice_UNegToZero`` for moving the upper bound of a
           slice by 1 unit e.g. ``x[1:5]`` becomes ``x[1:4]``.


0.6.1
~~~~~

    - Added explicit tests for ``argparse`` cli options.
    - Added mechanism to sort reporting mutations by source file, then line number, then column
      number.

0.6.0
~~~~~

    - Including ``pytest`` in the installation requirements. Technically, any test runner can
      be used but with all base package assumptions being built around ``pytest`` this feels
      like the right assumption to call out as an install dependency. It is the default behavior.
    - Updated ``controller`` for test file exclusion to explicitly match prefix or suffix cases
      for ``"test_"`` and ``"_test"`` per ``pytest`` conventions.
    - Changed error and unknown status results to console color as yellow instead of red.
    - Added multiple invariant property tests, primarily to ``controller`` and ``cache``.
    - Added ``hypothesis`` to the test components of ``extras_require``.
    - Moved to ``@property`` decorators for internal class properties that should only
      be set at initialization, may add custom ``setters`` at a later time.
    - Fixed a zero-division bug in the ``cli`` when reporting coverage percentage.

0.5.0
~~~~~

    - Addition of ``optimizers``, including the new class ``CoverageOptimizer``.
    - This optimizer restricts the full sample space only to source locations that are marked
      as covered in the ``.coverage`` file. If you have a ``pytest.ini`` that includes
      the ``--cov=`` command it will automatically generate during the clean-trial run.


0.4.2
~~~~~

    - More behind the scenes maintenance: updated debug level logging to include source file
      names and line numbers for all visit operations and separated colorized output to a new
      function.

0.4.1
~~~~~

    - Updated the reporting functions to return colorized display results to CLI.

0.4.0
~~~~~

    - Added new mutation support for:
        1. ``AugAssign`` in AST e.g. ``+= -= *= /=``.
        2. ``Index`` substitution in lists e.g. take a positive number like ``i[1]`` and
           mutate to zero and a negative number e.g. ``i[-1] i[0]``.

    - Added a ``desc`` attribute to ``transformers.MutationOpSet`` that is used in the
      cli help display.
    - Updated the cli help display to show the description and valid members.

0.3.0
~~~~~

    - Added new mutation support for ``NameConstant`` in AST.
    - This includes substitutions for singleton assignments such as: ``True``, ``False``,
      and ``None``.
    - This is the first non-type mutation and required adding a ``readonly`` parameter
      to the ``transformers.MutateAST`` class. Additionally, the type-hints for the
      ``LocIndex`` and ``MutationOpSet`` were updated to ``Any`` to support
      the mixed types. This was more flexible than a series of ``overload`` signatures.

0.2.0
~~~~~

    - Added new compare mutation support for:
        1. ``Compare Is`` mutations e.g. ``is, is not``.
        2. ``Compare In`` mutations e.g. ``in, not in``.

0.1.0
~~~~~

    - Initial release!
    - Requires Python 3.7 due to the ``importlib`` internal references for manipulating cache.
    - Run mutation tests using the ``mutatest`` command line interface.
    - Supported operations:

        1. ``BinOp`` mutations e.g. ``+ - / *`` including bit-operations.
        2. ``Compare`` mutations e.g. ``== >= < <= !=``.
        3. ``BoolOp`` mutations e.g. ``and or``.



---
File: /CONTRIBUTING.rst
---

Contributing
============

Up top, thanks for considering a contribution!
New features that align to the vision are welcome.
You can either open an issue to discuss the idea first, or if you have working code,
submit a pull-request.

Vision
------

The goal of ``mutatest`` is to provide a simple tool and API for mutation testing.
The top level priorities for the project are:

1. Collect useful mutation patterns without modifying the target source code.
2. Make it fast.

Open questions I'm working through on the design:

1. Fancy test selection? If there is a way to not only test coverage, but only select tests based
   on the mutation location (a form of "who tests what") that could make trials more efficient.

2. Local database? Keeping a local database of mutations and trial results would allow for re-running
   failed mutations quickly. Providing the ability to log false positives to skip on future samples
   would also be valuable.

3. Clustered mutations? There could be room for specifying a number of mutations to run simultaneously.

4. More API options? If you add two Genomes together should it create a GenomeGroup automatically?

5. More reporting options? HTML etc.


Development Guidelines
----------------------

The following guidelines are used in the style formatting of this package. Many are enforced through
``pre-commit`` Git hooks and in the test running configuration of ``tox``.

Development environment setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is how to get up and running for development on ``mutatest``. Referenced tools are included
in the development dependencies as part of the set up procedure.

1. Fork this repo, then clone your fork locally.
2. Create a new Python virtual environment using Python 3.7 and activate it.
3. Change to the local directory of your clone. All commands are run in the top level directory
   where the ``setup.py`` file is located.
4. Install ``mutatest`` in edit mode with all development dependencies using ``pip``.

.. code-block:: bash

    $ pip install -e .[dev]


5. Run a clean ``tox`` trial to ensure you're starting from a correct installation:

.. code-block:: bash

    $ tox

    # expected output ...

    py37: commands succeeded
    lint: commands succeeded
    typing: commands succeeded
    pypi-description: commands succeeded
    manifest: commands succeeded
    help: commands succeeded
    congratulations :)

6. Install ``pre-commit`` for the cloned repo. This ensures that every commit runs the
   formatting checks including ``black`` and ``flake8``.

.. code-block:: bash

    $ pre-commit install

7. Start developing!
8. Run ``tox`` one more time before you open the PR to make sure your functionality passes the
   original tests (and any new ones you have added).


Style: all files
~~~~~~~~~~~~~~~~

    - Generally hard-wrap at 100 characters for all files, including text files or RST.
    - Prefer RST over markdown or plaintext for explanations and outputs.
    - Accept the edits from the ``pre-commit`` configuration e.g. to trim trailing
      whitespace.


Style: Package Python code
~~~~~~~~~~~~~~~~~~~~~~~~~~

Many of these points are automated with ``pre-commit`` and the existing configuration settings
for ``black`` and ``flake8``. In general:


    - Use ``isort`` for ordering ``import`` statements in Python files.
    - Run ``black`` for formatting all Python files.
    - Use "Google Style" doc-string formatting for functions.
    - Type-hints are strictly enforced with ``mypy --strct``.
    - Adhere to PEP-8 for naming conventions and general style defaults.
    - All code is hard-wrapped at 100 characters.
    - If you are adding a new development tool instead of a feature, prefix the module name
      with an underscore.
    - Provide justification for any new install requirements.
    - All tests are stored in the ``tests/`` directory.
    - Accept the edits from the ``pre-commit`` configuration.


Style: Test Python code
~~~~~~~~~~~~~~~~~~~~~~~

``Pytest`` is used to manage unit tests, and ``tox`` is used to run various environment
tests. ``Hypothesis`` is used for property testing in addition to the unit tests.
If you are adding a new feature ensure that tests are added to cover the functionality.
Some style enforcing is relaxed on the test files:

    - Use ``isort`` for ordering ``import`` statements in Python files.
    - Run ``black`` for formatting all Python files.
    - Use "Google Style" doc-string formatting for functions, though single-line descriptions can be
      appropriate for unit test descriptions.
    - Test files are all in the ``mutatest/tests/`` directory so tests are distributed with the package.
    - Tests do not require type-hints for the core test function or fixtures. Use as appropriate to
      add clarity with custom classes or mocking.
    - Prefer to use ``pytest`` fixtures such as ``tmp_path`` and ``monkeypatch``.
    - All test files are prefixed with ``test_``.
    - All test functions are prefixed with ``test_`` and are descriptive.
    - Shared fixtures are stored in ``tests/conftest.py``.
    - Accept the edits from the ``pre-commit`` configuration.


Commits
~~~~~~~

    - Use descriptive commit messages in "action form". Messages should be read as, "If applied,
      this commit will... <<your commit message>>" e.g. "add tests for coverage of bool_op visit".
    - Squash commits as appropriate.



---
File: /README.rst
---

``mutatest``: Python mutation testing
==========================================

|  |py-versions| |license| |ci-azure| |docs| |coverage| |black|
|  |pypi-version| |pypi-status| |pypi-format| |pypi-downloads|
|  |conda-version| |conda-recipe| |conda-platform| |conda-downloads|


Are you confident in your tests? Try out ``mutatest`` and see if your tests will detect small
modifications (mutations) in the code. Surviving mutations represent subtle changes that are
undetectable by your tests. These mutants are potential modifications in source code that continuous
integration checks would miss.


Features
---------

    - Simple command line tool with `multiple configuration options <https://mutatest.readthedocs.io/en/latest/commandline.html>`_.
    - Built on Python's Abstract Syntax Tree (AST) grammar to ensure `mutants are valid <https://mutatest.readthedocs.io/en/latest/mutants.html>`_.
    - `No source code modification <https://mutatest.readthedocs.io/en/latest/install.html#mutation-trial-process>`_,
      only the ``__pycache__`` is changed.
    - Uses ``coverage`` to create `only meaningful mutants <https://mutatest.readthedocs.io/en/latest/commandline.html#coverage-filtering>`_.
    - Built for efficiency with `multiple running modes <https://mutatest.readthedocs.io/en/latest/commandline.html#selecting-a-running-mode>`_
      and `random sampling of mutation targets <https://mutatest.readthedocs.io/en/latest/commandline.html#controlling-randomization-behavior-and-trial-number>`_.
    - Capable of running `parallel mutation trials <https://mutatest.readthedocs.io/en/latest/commandline.html#parallelization>`_
      with multiprocessing on Python 3.8.
    - Flexible enough to run on a `whole package <https://mutatest.readthedocs.io/en/latest/commandline.html#auto-detected-package-structures>`_
      or a `single file <https://mutatest.readthedocs.io/en/latest/commandline.html#specifying-source-files-and-test-commands>`_.
    - Includes an `API for custom mutation controls <https://mutatest.readthedocs.io/en/latest/modules.html>`_.
    - Tested on Linux, Windows, and MacOS with `Azure pipelines <https://dev.azure.com/evankepner/mutatest/_build/latest?definitionId=1&branchName=master>`_.
    - Full strict static type annotations throughout the source code and the API.

Install
-------

Install from `PyPI <https://pypi.org/project/mutatest/>`_:

.. code-block:: bash

    $ pip install mutatest

Install from `conda-forge <https://anaconda.org/conda-forge/mutatest>`_:

.. code-block:: bash

    $ conda install -c conda-forge mutatest


Example Output
--------------

This is an output example running mutation trials against the
`API Tutorial example folder <https://mutatest.readthedocs.io/en/latest/api_tutorial/api_tutorial.html>`_
example folder.

.. code-block:: bash

    $ mutatest -s example/ -t "pytest" -r 314

    Running clean trial
    2 mutation targets found in example/a.py AST.
    1 mutation targets found in example/b.py AST.
    Setting random.seed to: 314
    Total sample space size: 2
    10 exceeds sample space, using full sample: 2.

    Starting individual mutation trials!
    Current target location: a.py, LocIndex(ast_class='BinOp', lineno=6, col_offset=11, op_type=<class '_ast.Add'>)
    Detected mutation at example/a.py: (6, 11)
    Detected mutation at example/a.py: (6, 11)
    Surviving mutation at example/a.py: (6, 11)
    Break on survival: stopping further mutations at location.

    Current target location: b.py, LocIndex(ast_class='CompareIs', lineno=6, col_offset=11, op_type=<class '_ast.Is'>)
    Detected mutation at example/b.py: (6, 11)
    Running clean trial

    Mutatest diagnostic summary
    ===========================
     - Source location: /home/user/Github/mutatest/docs/api_tutorial/example
     - Test commands: ['pytest']
     - Mode: s
     - Excluded files: []
     - N locations input: 10
     - Random seed: 314

    Random sample details
    ---------------------
     - Total locations mutated: 2
     - Total locations identified: 2
     - Location sample coverage: 100.00 %


    Running time details
    --------------------
     - Clean trial 1 run time: 0:00:00.348999
     - Clean trial 2 run time: 0:00:00.350213
     - Mutation trials total run time: 0:00:01.389095

    Trial Summary Report:

    Overall mutation trial summary
    ==============================
     - DETECTED: 3
     - SURVIVED: 1
     - TOTAL RUNS: 4
     - RUN DATETIME: 2019-10-17 16:57:08.645355

    Detected mutations:

    DETECTED
    --------
     - example/a.py: (l: 6, c: 11) - mutation from <class '_ast.Add'> to <class '_ast.Sub'>
     - example/a.py: (l: 6, c: 11) - mutation from <class '_ast.Add'> to <class '_ast.Mod'>
     - example/b.py: (l: 6, c: 11) - mutation from <class '_ast.Is'> to <class '_ast.IsNot'>

    Surviving mutations:

    SURVIVED
    --------
     - example/a.py: (l: 6, c: 11) - mutation from <class '_ast.Add'> to <class '_ast.Mult'>


Documentation
-------------

For full documentation, including installation, CLI references, API references, and tutorials,
please see https://mutatest.readthedocs.io/en/latest/.
The project is hosted on PyPI at https://pypi.org/project/mutatest/.


Bugs/Requests
-------------

Please use the `GitHub issue tracker <https://github.com/EvanKepner/mutatest/issues>`_ to submit bugs
or request features.
See the `Contributing Guidelines <https://mutatest.readthedocs.io/en/latest/contributing.html>`_ if you
are interested in submitting code in the form of pull requests.

ChangeLog
---------

Consult the `Changelog <https://mutatest.readthedocs.io/en/latest/changelog.html>`_ page for fixes
and enhancements of each version.

License
-------

Copyright Evan Kepner 2018-2020.

Distributed under the terms of the `MIT <https://github.com/pytest-dev/pytest/blob/master/LICENSE>`_
license, ``mutatest`` is free and open source software.

.. |py-versions| image:: https://img.shields.io/pypi/pyversions/mutatest?color=green
    :target: https://pypi.org/project/mutatest/
    :alt: Python versions
.. |license| image:: https://img.shields.io/pypi/l/mutatest.svg
    :target: https://pypi.org/project/mutatest/
    :alt: License
.. |pypi-version| image:: https://badge.fury.io/py/mutatest.svg
    :target: https://pypi.org/project/mutatest/
    :alt: PyPI version
.. |pypi-status| image:: https://img.shields.io/pypi/status/mutatest.svg
    :target: https://pypi.org/project/mutatest/
    :alt: PyPI status
.. |pypi-format| image:: https://img.shields.io/pypi/format/mutatest.svg
    :target: https://pypi.org/project/mutatest/
    :alt: PyPI Format
.. |pypi-downloads| image:: https://pepy.tech/badge/mutatest
    :target: https://pepy.tech/project/mutatest
    :alt: PyPI Downloads
.. |ci-travis| image:: https://travis-ci.org/EvanKepner/mutatest.svg?branch=master
    :target: https://travis-ci.org/EvanKepner/mutatest
    :alt: TravisCI
.. |ci-azure| image:: https://dev.azure.com/evankepner/mutatest/_apis/build/status/EvanKepner.mutatest?branchName=master
    :target: https://dev.azure.com/evankepner/mutatest/_build/latest?definitionId=1&branchName=master
    :alt: Azure Pipelines
.. |docs| image:: https://readthedocs.org/projects/mutatest/badge/?version=latest
    :target: https://mutatest.readthedocs.io/en/latest/?badge=latest
    :alt: RTD status
.. |coverage| image:: https://codecov.io/gh/EvanKepner/mutatest/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/EvanKepner/mutatest
    :alt: CodeCov
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black
.. |conda-recipe| image:: https://img.shields.io/badge/recipe-mutatest-green.svg
    :target: https://anaconda.org/conda-forge/mutatest
    :alt: Conda recipe
.. |conda-version| image:: https://img.shields.io/conda/vn/conda-forge/mutatest.svg
    :target: https://anaconda.org/conda-forge/mutatest
    :alt: Conda version
.. |conda-platform| image:: https://img.shields.io/conda/pn/conda-forge/mutatest.svg
    :target: https://anaconda.org/conda-forge/mutatest
    :alt: Conda platforms
.. |conda-azure| image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/mutatest-feedstock?branchName=master
    :target: https://anaconda.org/conda-forge/mutatest
    :alt: Conda azure status
.. |conda-downloads| image:: https://img.shields.io/conda/dn/conda-forge/mutatest.svg
    :target: https://anaconda.org/conda-forge/mutatest
    :alt: Conda downloads

