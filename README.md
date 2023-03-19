# Alzheimer Model
Machine learning model for reccognizing amyloid deposits on the brain cortex


## REQUIREMENTS

## Pyenv

- Install:

```bash
$ git clone https://github.com/pyenv/pyenv.git ~/.pyenv
$ cd ~/.pyenv && src/configure && make -C src
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
$ echo 'eval "$(pyenv init --path)"' >> ~/.profile
```
Log out and log in again

- Use
```bash
$ pyenv install 3.X.X
$ pyenv local 3.X.X
```
This creates a `.python-version` file and fi the version for everybody else.

If another person downloads the repo (having pyenv installed) and run `python --version` the version should see somthing like this:
```bash
$ python --version
pyenv: version `3.X.X' is not installed (set by /home/.../repo-name/.python-version)
```
Then, you can simply install the required version by running `pyenv install` or `pyenv install 3.X.X`

## Virtual Environment

- Create one: Once you have the python version ser un correctly, you can create the virtual environment. 

```bash
$ python --version  # check version matches the required version
$ python -m venv .venv
```
- Activate it:
```bash
$ source .venv/bin/activate
```

## Poetry 
- Install poetry: https://python-poetry.org/docs/#installation
- Install dependencies:
```bash
$ poetry install
```