# CE-OCL
 
Repo to reproduce results of the manuscript **Counterfactual Explanations Using Optimization With Constraint Learning**. 
 
 
 ## repo structure
     .
    ├── case study/      # resources to reproduce results of the two case studies in Section 3 and Appendix D  
    |   ├── data/
    |   |   └── ...
    |   ├── results/
    |   |   └── ...
    |   ├── CE-OCL Case Study - Statlog (German Credit).ipynb
    |   └── CE-OCL Case Study - Statlog (Heart).ipynb
    |
    ├── numerical experiments/     # resources to reproduce results of Section 3 and Appendix C
    |   ├── data/
    |   |   └── ...
    |   ├── results/
    |   |   └── ...
    |   ├── CE-OCL_DiCE_results.py
    |   ├── Datasets.py
    |   └── evaluate_carla_methods.py
    |
    ├── src/   # source files and helper functions
    |   └── ...
    |   
    ├── requirements.txt
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
