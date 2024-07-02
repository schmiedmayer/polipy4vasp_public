.PHONY=all \
       install \
       install-dev \
       install-dev-all \
       uninstall \
       rebuild \
       clean-build \
       doc \
       clean-doc \
       check-docstyle \
       test \
       clean-test

all: install-dev-all

install:
	pip install --user .

install-dev:
	pip install --prefix=$(shell python3 -m site --user-base) -e .

install-dev-all:
	pip install --prefix=$(shell python3 -m site --user-base) -e .[docs,tests]

uninstall: clean-build
	pip uninstall polipy4vasp
	$(RM) -r polipy4vasp.egg-info/

rebuild: clean-build
	python setup.py build_ext && cp build/lib*/vasa* .

clean-build:
	$(RM) -r build
	$(RM) vasa.*.so

doc: check-docstyle
	cd docs && sphinx-apidoc -o . ../polipy4vasp &&	$(MAKE) html

clean-doc:
	cd docs && $(MAKE) clean && $(RM) modules.rst polipy4vasp.rst

check-docstyle:
	-pydocstyle --convention=numpy polipy4vasp/*.py

test:
	pytest -v --cov=polipy4vasp --cov-report term --cov-report html

clean-test:
	$(RM) -r .pytest_cache htmlcov .coverage polipy4vasp/.coverage polipy4vasp/__pycache__ polipy4vasp/tests/__pycache__

black-check:
	black --check .

black-diff:
	black --color --diff .

black-format:
	black .
