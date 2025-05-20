# magnetic-materials-2d

Here is a quick tutorial on how to locally install the package.

## How to install `magnetic-materials-2d` locally

`cd` into the project directory:

```bash
cd magnetic-materials-2d
```

Create and activate a new conda environment:

```bash
conda create -n magnetic-materials-2d_env python=<max_python_version>
conda activate magnetic-materials-2d_env
```

### Method 1: Install your package with dependencies sourced from pip

It's simple. The only command required is the following:

```bash
pip install -e .
```

> The above command will automatically install the dependencies listed in `requirements/pip.txt`.

### Method 2: Install your package with dependencies sourced from conda

If you haven't already, ensure you have the conda-forge channel added as the highest priority channel.

```bash
conda config --add channels conda-forge
```

Install the dependencies listed under `conda.txt`:

```bash
conda install --file requirements/conda.txt
```

Then install your Python package locally:

```bash
pip install -e . --no-deps
```

> `--no-deps` is used to avoid installing the dependencies in `requirements/pip.txt` since they are already installed in the previous step.

## Verify your package has been installed

Verify the installation:

```bash
pip list
```

Great! The package is now importable in any Python scripts located on your local machine. For more information, please refer to the Level 4 documentation at [https://billingegroup.github.io/scikit-package/](https://billingegroup.github.io/scikit-package/).
