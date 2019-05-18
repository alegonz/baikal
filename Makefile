setup_dev:
	pip3 install -e .[dev,viz]

test:
	pytest -s -vv tests/
