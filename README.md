# MECH5720 Major: Schlieren Imaging

This repository contains the code used for the Background Orientated Schlieren processing code.

## Installations (Linux)

Create a venv on Python version `3.14` with `python3.14 venv .venv`.

Activate the venv with `source .venv/bin/activate`.

Install requirements with `pip install -r requirements.txt`.

## Running

`src/find_fix_pattern.py` can be used to determine the fixed-pattern using black-frame captures in `fixed_pattern`.

`src/noise_gen.py` can be used to generate noise patterns to use as the background.
