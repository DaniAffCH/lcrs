#!/bin/bash

git clone --recurse-submodules https://github.com/mees/calvin.git
pip install -e .
cd calvin/calvin_env/tacto
pip install -e .
cd ..
pip install -e .
cd ../calvin_models
pip install -e .
cd ../..
