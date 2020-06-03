.PHONY: clean clean-build clean-pyc install install-dev uninstall lint format conda-install conda-remove create-env remove-env check-env

# Verbose mode. Call make with V=1 to enable targets log: e.g. 'make install V=1'
$(V).SILENT:

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ENV_NAME := $(shell sed -nE 's/^.*name:\s*(.*)$$/\1/p' .template_environment.yml)

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Remove build and compiled python artifacts
clean: clean-build clean-pyc

## Remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

## Remove python compiled artifacts
clean-pyc:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Install the package to the active Python's site-packages
install: clean
	pip install .

## Install the package in editable form for development
install-dev: clean
	pip install -e .

## Uninstall the package
uninstall:
	pip uninstall -y ml_project
	$(MAKE) clean

## Lint using flake8
lint:
	flake8 --exit-zero src

## Format using black
format:
	black src

## Install Miniconda package manager
conda-install:
ifeq (True,$(HAS_CONDA))
	echo "Conda is already installed. Skipping installation"
else
	scripts/install_conda.sh
endif

## Remove Miniconda package manager
conda-remove:
	echo "Due to how conda works it is not possible to entirely remove it automatically."
	echo "Here are the necessary steps:"
	echo "1."
	echo "    1.a Use the provided script, by sourcing it:"
	echo "        source scripts/remove_conda.sh"
	echo "    1.b Or use manual remotion:"
	echo "        * Deactivate all environments (including base): conda deactivate"
	echo "        * Remove directories: rm -r <conda install directory> .conda"
	echo "2."
	echo "    Remove modifications from .bashrc file"
	echo ""

## Set up conda environment: pytorch, cudatoolkit and torchvision versions must be defined by the user as they depend on the installed cuda runtime version
create-env:
ifeq (True,$(HAS_CONDA))
ifdef pytorch
ifdef cudatoolkit
	echo "Chosen dependecies versions:"
	echo "  pytorch version: $(pytorch)"
	echo "  cudatoolkit version: $(cudatoolkit)"
	echo ""
	echo "Creating environment.yml file (from chosen dependencies and template file)"
	echo "Installing environment $(ENV_NAME) ..."
	sed -E 's/<PYTORCH_VERSION>.*$$/$(pytorch)/;s/<CUDATOOLKIT_VERSION>.*$$/$(cudatoolkit)/;/^# WARNING/d' .template_environment.yml > environment.yml && conda env create --file environment.yml
	echo ""
	echo "Environment created. You can check that pytorch can connect with the gpu with 'make check-env'"
	echo ""
	echo "The package must be installed with 'make install' or 'make install-dev'"
else
	echo "cudatoolkit version must be defined. E.g.: 'make create-env cudatoolkit=10.0 ...'"
endif
else
	echo "pytorch version must be defined. E.g.: 'make create-env pytorch=1.4.0 ...'"
endif
else
	echo "Error: conda not found. Install with 'make conda-install'"
endif

## Remove conda environment
remove-env:
	conda env remove --name $(ENV_NAME)

## Test that project and python environment are setup correctly
check-env:
	python scripts/check_environment.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	echo "$$(tput bold)Available rules:$$(tput sgr0)"
	echo
	sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
