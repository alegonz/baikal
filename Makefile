venv: venv/bin/activate

venv/bin/activate:
	@echo Using $(shell python3 --version) at $(shell which python3)
	python3 -m venv venv

clean:
	rm -rf build dist baikal.egg-info pip-wheel-metadata

setup_dev: venv
	. venv/bin/activate; \
	pip install -U pip; \
	pip install -e .[dev,viz]; \
	pip install pre-commit; \
	pre-commit install

test:
	. venv/bin/activate; \
	pytest -s -vv tests/

test-cov:
	. venv/bin/activate; \
	pytest -s -vv --cov-config .coveragerc --cov=baikal tests/

upload-cov:
	. venv/bin/activate; \
	codecov --token=${CODECOV_TOKEN}

type-check:
	. venv/bin/activate; \
	mypy --ignore-missing-imports --allow-redefinition baikal/ tests/

wheel: clean venv
	. venv/bin/activate; \
	pip install --upgrade setuptools wheel; \
	python setup.py sdist bdist_wheel

upload: dist venv
	. venv/bin/activate; \
	pip install --upgrade twine; \
	twine check dist/*; \
	twine upload dist/*
