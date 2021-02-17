# Almiky

## Python library for data hiding in images

### Download and active enviroment

Config virtualenv:

```bash
$ git clone https://gitlab.udg.co.cu/dprocessing/almiky.git
$ python3 -m venv env
$ source env/bin/activate
```

### For local development

Install requirements:

```bash
$ cd almiky/
$ pip install -r requirements.txt
```

Run tests:

```bash
$ source run-tests.sh
```

Code quality:

```bash
$ flake8 almiky
```

### Packaging project

Build the package:

```bash
$ pip install setuptools wheel
$ cd almiky/
$ python3 setup.py sdist bdist_wheel
```

### Docs

Build the doc:

```bash
$ cd almiky/docs/
$ make html
```
