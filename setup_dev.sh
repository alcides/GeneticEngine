#!/bin/bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
pre-commit install
