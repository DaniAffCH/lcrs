# LCRS
Language Conditioned Robot Skills

# Installation
This repository is based on Calvin framework, use the flag `--recursive` to download the framework properly:

1. **Clone Repository:**
   ```bash
   git clone --recursive https://github.com/DaniAffCH/lcrs.git
   export LCRS_ROOT=$(pwd)/lcrs
2. **Install lcrs Repository:**
   To install the repository, we strongly recommend using the provided conda environment to manage pip dependencies.
   ```bash
   cd $LCRS_ROOT
   conda env create -f environment.yml
   source activate lcrs_venv
   sh install.sh
   # Optionally setup jupyternotebook kernel
   python -m ipykernel install --user --name=lcrs_venv
   ```

4. **Configure W&B logger**
   Modify the `$LCRS_ROOT/conf/logger/wandb.yaml` file by setting the `project` field with the project name of your W&B workspace and the `entity` field with your W&B username.
5. **Download data**
You can download the dataset with the scripts provided in [dataset](./dataset/) directory.

## Troubleshooting

If you get errors during conda environment creation regarding pyhash/setuptools, activate the environment anyway and then install setuptools manually:
```bash
pip install setuptools==57.4.0
```

If you encounter errors during build wheels for MulticoreTSNE, try downgrading the cmake version to e.g. 3.18.4: 
```bash
pip install cmake==3.18.4
```

Don't worry about the errors like `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. [...]`

# Dataset download

You can download the calvin datasets as follows:

```bash
cd $LCRS_ROOT/dataset
sh download_data.sh D | ABC | ABCD | debug
```

Alternativley, you the dataset gets automatically downloaded when using the example [training.ipynb](./training.ipynb).

# Training
```bash
python lcrs/training.py datamodule.root_data_dir=/path/to/dataset
```
or via the python notebook [training.ipynb](./training.ipynb) (remember to set the correct conda kernel before execution).
