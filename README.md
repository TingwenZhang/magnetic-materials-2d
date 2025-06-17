# magnetic-materials-2d

Predict the formation energy and magnetic moment of some 2D magentic materials.

## Project Structure

```
./src
├── magnetic_materials_2d
│   ├── **init**.py
│   ├── **pycache**
│   │   ├── **init**.cpython-313.pyc
│   │   ├── functions.cpython-313.pyc
│   │   ├── hyper_search.cpython-313.pyc
│   │   └── utils.cpython-313.pyc
│   ├── data
│   │   ├── README.rtf
│   │   ├── **pycache**
│   │   │   └── dictionaries.cpython-313.pyc
│   │   ├── dictionaries.py
│   │   └── magneticmoment_Ef_data.csv
│   ├── functions.py
│   ├── hyper_search.py
│   └── utils.py
├── magnetic_materials_2d.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
└── notebooks
├── ML_2D_exercises.ipynb
└── ML_2D_working.ipynb
```

- `/src/magnetic_materials_2d` contains all python modules.
- `/src/magnetic_materials_2d/data` contains information about the descriptors.
- `/src/notebooks` is where the running notebooks are; they will utilize the python modules and data.

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

**Use pull-requests or issues to share any improvements to the documentation or codebase.**
