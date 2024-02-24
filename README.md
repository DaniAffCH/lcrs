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
   ```

4. **Configure W&B logger**
   Modify the `$LCRS_ROOT/conf/logger/wandb.yaml` file by setting the `project` field with the project name of your W&B workspace and the `entity` field with your W&B username.
5. **Download data**
You can download the dataset with the scripts provided in [dataset](./dataset/) directory.

## Troubleshooting
If you encounter errors during build wheels for MulticoreTSNE, try downgrading the cmake version to e.g. 3.18.4: 
```bash
pip install cmake==3.18.4
```
Don't worry about the errors like `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. [...]`

# Hack for language-only rollout
Modify the rollout within calvin
```bash
calvin/calvin_models/calvin_agent/rollout/rollout.py
```
by adding a `continue` in the else block of line 308.

```python
   else:
      # goal image is last step of the episode
      continue
      goal = {
          "rgb_obs": {k: v[i, -1].unsqueeze(0).unsqueeze(0) for k, v in rgb_obs.items()},  # type: ignore
          "depth_obs": {k: v[i, -1].unsqueeze(0).unsqueeze(0) for k, v in depth_obs.items()},  # type: ignore
          "robot_obs": state_obs[i, -1].unsqueeze(0).unsqueeze(0),
      }
```

# Training
```bash
python lcrs/training.py datamodule.root_data_dir=/path/to/dataset
