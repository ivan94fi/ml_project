#!/bin/bash
installer="Miniconda3-py37_4.8.2-Linux-x86_64.sh"
conda_home=$HOME/miniconda3

if [ ! -e /tmp/$installer ];
then
    echo "downloading the installer.."
    wget https://repo.anaconda.com/miniconda/$installer -O /tmp/$installer
fi

echo "Miniconda install location (press enter to accept default):"
printf "[%s] >>> " "$conda_home"
read -r custom_home
if [ "$custom_home" != "" ]; then
    conda_home=$custom_home
fi

echo "installing in $conda_home.."
bash /tmp/$installer -b -p $conda_home
source $conda_home/bin/activate
conda init

echo " "
echo "installation concluded"
echo "restart the shell for changes to take effect"
