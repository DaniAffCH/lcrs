# lcrs
Language Conditioned Robot Skills

## Installation 
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
4. 
