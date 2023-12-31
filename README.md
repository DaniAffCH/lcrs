# LCRS
Language Conditioned Robot Skills

# Installation
This repository requires Calvin to be installed.
To integrate the LCRS module into Calvin, navigate to the calvin root and clone this repository:

1. **Clone Repository:**
   ```bash
   git clone https://github.com/DaniAffCH/lcrs.git
   export LCRS_ROOT=$(pwd)/lcrs
2. **Install lcrs Repository:**
   ```bash
   cd $LCRS_ROOT && sh install.sh
3. **Configure W&B logger**
   Modify the `$LCRS_ROOT/conf/logger/wandb.yaml` file by setting the `project` field with the project name of your W&B workspace and the `entity` field with your W&B username.
4. **Download data**
You can download the dataset with the scripts provided in [dataset](./dataset/) directory.

## Troubleshooting
If you encounter errors during build wheels for MulticoreTSNE, try downgrading the cmake version to e.g. 3.18.4: 
```bash
pip install cmake==3.18.4
```
(the commit 79fc18548e2fb488399ec7fe0b9905bc7296ca63 solved the issue)

Don't worry about the errors like `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. [...]`

# Training
```bash
python lcrs/training.py datamodule.root_data_dir=/path/to/dataset datamodule/datasets=vision_lang_shm
