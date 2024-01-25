#!/bin/bash

venv_name="lcrsVenv"

if command -v python3 &> /dev/null; then
    if [ ! -d "$venv_name" ]; then
        python3 -m venv "$venv_name"
    fi

    source "$venv_name/bin/activate"
    pip install -e .
    cd calvin
    sudo chmod +x ./install.sh
    ./install.sh
else
    echo "Error: Python3 is not installed. Please install Python3 before running this script."
    exit 1
fi

