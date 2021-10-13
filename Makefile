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
