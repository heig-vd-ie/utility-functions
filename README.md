Swiss-topo-API
==============

_API used to download data from [swiss geo admin](https://www.bfe.admin.ch/bfe/en/home/supply/digitalization-and-geoinformation/programming-interfaces/geoadmin-api.html)._


### Step 1. Install pipx on wsl (if not installed)

```bash
sudo apt update
sudo apt install pipx
pipx ensurepath 
pipx ensurepath --force
```

### Step 2. Install python 3.12 on wsl (if not installed)

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12
sudo apt install python3.12-venv
```
Reference: [Tutorial is the following link](https://www.linuxtuto.com/how-to-install-python-3-12-on-ubuntu-22-04/)

> [!IMPORTANT]
> If psycopg-c installation raise the [error](https://stackoverflow.com/questions/77727508/problem-installing-psycopg2-for-python-venv-through-poetry): _psycopg-c (3.1.18) not supporting PEP 517 builds_

```bash
sudo apt install libpq-dev gcc
sudo apt install python3.12-dev
```
### Step 4. Install Poetry (of not installed)

```bash
pipx install poetry
```
### Step 5. Add direnv

#### 5.1 Install direnv (if not installed)

```bash
sudo apt install direnv
```

#### 5.2 Check which shell you use (bash advised)

```bash
echo $0
```

#### 5.3 Add direnv hook in bash config file

```bash
nano ~/.bashrc
```

If you don't use bash, please check [direnv doc](https://direnv.net/docs/hook.html).

At the end of the file add the following row:

```bash
eval "$(direnv hook bash)"
```

#### Step 5.4 Create .envrc file and add the following row

```bash
export PYTHONPATH=$(pwd)/src
```

#### Step 5.5 Allow terminal to use `direnv`

```bash
direnv allow
```

You can check if the environment variable has been created (it should return the src folder path).

```bash
echo $PYTHONPATH
```

### Step 6.Create pyproject.toml file (if needed)
```bash
poetry init
```

### Step 7. Create virtual environment (`.venv`)

```bash
python3.12 -m venv .venv
poetry env use .venv/bin/python3.12
```

### Step 8. Install packages in  `.venv` environement

```bash
poetry install
```


### Step 9. Install Julia API 
#### 9.1 Download Julia (of not installed)
```bash
curl -fsSL https://install.julialang.org | sh
```
#### 9.2 Open a Julia session of the current project
```bash
julia --project
```
#### 9.3 Inside the opened Julia session, install the required packages specified in Project.toml
```bash
]
status
instantiate
```
