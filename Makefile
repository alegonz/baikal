SHELL:=/bin/bash
SKLEARN_NIGHTLY_URL:=https://sklearn-nightly.scdn8.secure.raxcdn.com
ifndef SKLEARN_VERSION
override SKLEARN_VERSION=unspecified
endif

venv: venv/bin/activate

venv/bin/activate:
	@echo Using $(shell python3 --version) at $(shell which python3)
	python3 -m venv venv

clean:
	rm -rf build dist baikal.egg-info pip-wheel-metadata

setup_dev: venv
	. venv/bin/activate && \
	pip install -U pip && \
	if [[ ${SKLEARN_VERSION} =~ [0-9\.] ]]; then \
        pip install scikit-learn==${SKLEARN_VERSION}; \
    elif [[ ${SKLEARN_VERSION} = "nightly" ]]; then \
        pip install --ignore-installed --pre -f ${SKLEARN_NIGHTLY_URL} scikit-learn; \
	fi && \
	pip install -e .[dev,viz] && \
	pip install pre-commit && \
	pre-commit install

test:
	. venv/bin/activate && \
	pytest -s -vv tests/

test-cov:
	. venv/bin/activate && \
	pytest -s -vv --cov-config .coveragerc --cov=baikal tests/

upload-cov:
	. venv/bin/activate && \
	codecov --token=${CODECOV_TOKEN}

type-check:
	. venv/bin/activate && \
	mypy --ignore-missing-imports baikal/ tests/

wheel: clean venv
	. venv/bin/activate && \
	pip install --upgrade setuptools wheel && \
	python setup.py sdist bdist_wheel

upload: dist venv
	. venv/bin/activate && \
	pip install --upgrade twine && \
	twine check dist/* && \
	twine upload dist/*
