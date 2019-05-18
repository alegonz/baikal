setup_dev:
	pip3 install -e .[dev]

test:
	pytest -s -vv tests/
