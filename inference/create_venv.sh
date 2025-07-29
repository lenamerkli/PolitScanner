if [ ! -d .venv ]; then
    /usr/bin/python3.12 -m venv .venv
fi

# install packages
.venv/bin/pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip
.venv/bin/pip3 install wheel==0.46.1 setuptools==79.0.0 flask==3.1.0 requests==2.32.3 tqdm==4.67.1 chromadb==1.0.7 certifi==2025.6.15
