# CE-OCL
 
 
 
 ## repo structure
     .
    ├── case study/         
    |   ├── data/
    |   |   └── ...
    |   ├── results/
    |   |   └── ...
    |   ├── CE-OCL Case Study - Statlog (German Credit).ipynb
    |   └── CE-OCL Case Study - Statlog (Heart).ipynb
    |
    ├── numerical experiments/     
    |   ├── data/
    |   |   └── ...
    |   ├── results/
    |   |   └── ...
    |   ├── CE-OCL_DiCE_results.py
    |   ├── Datasets.py
    |   └── evaluate_carla_methods.py
    |
    ├── src/   # helper functions needed for all notebooks
    |   └── ...
    |   
    ├── requiremsnts.txt
    └── README.md


## set up virtual environment

In your terminal, run: 

```
python3.8 -m venv venv

source venv/bin/activate

# check python version
python --version

# check installed packages
pip list

# install DiCE
pip install dice-ml

# install requirements
cd /Users/tabearober/Documents/Counterfactuals/CE-OCL
pip install -r requirements.txt

# install jupyter
pip install jupyter

# install Gurobi
visit https://www.gurobi.com/products/gurobi-optimizer/

# exit virtual env
deactivate
```
