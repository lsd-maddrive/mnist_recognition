#* Variables
PYTHON := python3
OS := $(shell uname -o)

#################################
# Poetry install/uninstall
#################################
#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) - --uninstall

.PHONY: poetry-install
poetry-install:
	poetry install -n

.PHONY: poetry-export
poetry-export:
	poetry export --without-hashes > requirements.core.txt

.PHONY: poetry-export-dev
poetry-export-dev:
	poetry export --without-hashes --dev > requirements.txt


.PHONY: tools-install
tools-install:
	poetry run pre-commit install
	poetry run nbdime config-git --enable

# Prepare project when you first time load it
prepare-project:
	poetry install
	poetry run pre-commit install

    ifeq ($(OS),GNU/Linux)
		chmod +x ./git_hook/git-hook.sh
		./git_hook/git-hook.sh
    else
        git_hook\git-hook.bat
    endif

#* Installation
.PHONY: install
install: poetry-install tools-install


#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf
	find . | grep -E "(.ipynb_checkpoints$$)" | xargs rm -rf
