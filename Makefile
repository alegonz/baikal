setup_dev:
	pip3 install -e .[dev,viz]

test:
	pytest -s -vv tests/

test-cov:
	pytest -s -vv --cov-config .coveragerc --cov=baikal tests/
