# CE-OCL
 
 
 
 ## repo structure
     .
    ├── data/         
    |   └── ...
    |
    ├── results/     # this directory will be automatically generated when running the notebooks
    |   └── ...
    |
    ├── ce_helpers.py   # helper functions needed for all notebooks
    ├── embed_mip.py    # addition to OptiCL that allows for enlarged data manifold region
    ├── CE-OCL Case Study.ipynb     # notebook to reproduce results of the case study (Section 4.1)
    ├── Evaluation CE-OCL.ipynb     # notebook to reproduce results of Section 4.2 (for CE-OCL)
    |   
    └── README.md


## set up virtual environment

:warning: adjust directories to the packages DiCE and OptiCL. Make sure that DiCE is the one we modified to also be able to handle SVM.

In your terminal, run: 

```
python3.8 -m venv venv

source venv/bin/activate

# check python version
python --version

# check installed packages
pip list

# install DiCE
cd /Users/tabearober/Documents/Counterfactuals/packages/DiCE
pip install -e .
#pip install dice-ml

# install OptiCL
# first navigate to the package, then install using pip
cd /Users/tabearober/Documents/Counterfactuals/packages/OptiCL 
pip install .

# install requirements
cd /Users/tabearober/Documents/Counterfactuals/CE-OCL
pip install -r requirements.txt

# install jupyter
pip install jupyter

# exit virtual env
deactivate
```
