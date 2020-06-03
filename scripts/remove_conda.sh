#!/bin/bash
conda_home=$HOME/miniconda3

echo "Unistalling conda"
echo "This script must be sourced, e.g. '. remove_conda.sh'."
echo ""

echo "If you installed conda on a custom path, please specify it, else press enter"
printf "[%s] >>> " "$conda_home"
read -r custom_home
if [ "$custom_home" != "" ]; then
    conda_home=$custom_home
fi

echo "Removing Miniconda installed at $conda_home"
conda deactivate
conda deactivate
rm -r $HOME/.conda $conda_home

echo " "
echo "Uninstall complete"
echo "Please manually remove modifications from your .bashrc file"
