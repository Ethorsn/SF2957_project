# SF2957 project

## Setup the env. (Linux)
Run in the following order
```bash
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh

./Anaconda3-5.1.0-Linux-x86_64.sh 

conda env create --file setup.yaml
```
for you to set up on your own machine, download anaconda accordingly (check IOS).

## Run script for generating pictures.
```bash 
# from root of repo.
mkdir ./report/figures ./data

source activate AIgym
python code/run_q_learning.py

```
