.PHONY: venv help test lint

.default: help

help:
	@echo "usage:"


ifeq (${VIRTUAL_ENV},)
  VENV_NAME = .venv
  VENV_BIN = ${VENV_NAME}/bin
  ${info Using ${VENV_NAME}}
else
  VENV_NAME = ${VIRTUAL_ENV}
  VENV_BIN = ${VENV_NAME}/bin
  ${info Using ${VENV_NAME}}
endif
ifeq (${VIRTUAL_ENV},)
  VENV_ACTIVATE = . ${VENV_BIN}/activate
else
  VENV_ACTIVATE = true
endif
PYTHON = ${VENV_BIN}/python


venv: ${VENV_NAME}/venv.created

${VENV_NAME}/venv.created:
	test -d ${VENV_NAME} || python -m venv ${VENV_NAME}
	@touch $@

${VENV_NAME}/dev.installed: setup.py setup.cfg requirements.txt
	${VENV_ACTIVATE}; python -m pip install -Ue .[dev]
	@touch $@

install-dev: venv ${VENV_NAME}/dev.installed

test: install-dev
	${VENV_ACTIVATE}; pytest --cov=ml_mastery  --cov-report=term-missing ml_mastery/tests

lint: install-dev
	${VENV_ACTIVATE}; pylint --rcfile=.pylintrc ml_mastery

