Introduction
============

Installation
------------

.. code:: bash

   make

Uninstall everything with ``make uninstall``.

Rebuilding the Fortran source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   make rebuild

Remove fortran build ``make clean-build``.

Running tests
^^^^^^^^^^^^^

.. code:: bash

   make test

Remove test results with ``make clean-test``.

Generate documentation
^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   make doc

Remove documentation with ``make clean-doc``.

Format code with *black*
^^^^^^^^^^^^^^^^^^^^^^^^

First, check whether *black* would reformat any files:

.. code:: bash

   make black-check

If yes, then review the changes:

.. code:: bash

   make black-diff

Finally, apply the suggested changes:

.. code:: bash

   make black-format
