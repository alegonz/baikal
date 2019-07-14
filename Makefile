clean:
	rm -rf build dist baikal.egg-info

setup_pkgs:
	pip3 install --upgrade setuptools wheel

setup_dev:
	pip3 install -e .[dev,viz]

test:
	pytest -s -vv tests/

test-cov:
	pytest -s -vv --cov-config .coveragerc --cov=baikal tests/

wheel: clean setup_pkgs
	python3 setup.py sdist bdist_wheel

upload: dist
	pip3 install --upgrade twine
	twine check dist/*
	twine upload dist/*
