check_code:
	@flake8 flower_classif/*.py

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__
	@rm -fr __pycache__
	@rm -fr build
	@rm -fr dist
	@rm -fr flower_classif-*.dist-info
	@rm -fr flower_classif.egg-info

all: clean install test check_code

install: clean wheel
	@pip3 install -U dist/*.whl

install_requirements:
	@pip3 install -r requirements.txt

wheel:
	@rm -f dist/*.whl
	@python3 setup.py bdist_wheel  # --universal if you are python2&3

test:
	@coverage run -m unittest tests/*.py
	@coverage report -m --omit=$(VIRTUAL_ENV)/lib/python*,flower_classif/*

