#!/bin-bash                                                                                                                                                                                                

sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
     libbz2-dev libreadline-dev libsqlite3-dev curl \
     libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL"

git clone git@github.com:trujicata/alzheimer-model.git
cd alzheimer-model
pyenv install
pyenv local
curl -sSL https://install.python-poetry.org | python3 -
python -m venv .venv
source .venv/bin/activate
poetry install
pip install -U -r requirements.gpu.txt
