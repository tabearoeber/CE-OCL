
import pandas as pd
import numpy as np
import ce_helpers
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
# from importlib import reload
# reload(ce_helpers)
import Datasets as DS



wd = '/Users/tabearober/Documents/Counterfactuals/CE-OCL/data/'
alg = 'rf'
dataset = DS.compas
results_path = '/Users/tabearober/Documents/Counterfactuals/CE-OCL/results/'


'''
---------DATA---------
'''

df, df_train, df_test, d, clf, df_factuals, model_master, encoder, scaler = dataset(wd, alg)

path = wd + '%s/CARLA_counterfactuals_%s' % (dataset.__name__, dataset.__name__)
with open(path, 'rb') as f:
    counterfactuals = pickle.load(f)


'''
---------EVALUATE---------
'''

methods = ['GrowingSpheres', 'FACE', 'AR']

validity = {'FACE':[],'AR':[],'GrowingSpheres':[]}
cat_prox = {'FACE':[],'AR':[],'GrowingSpheres':[]}
cont_prox = {'FACE':[],'AR':[],'GrowingSpheres':[]}
sparsity = {'FACE':[],'AR':[],'GrowingSpheres':[]}
cat_diver = {'FACE':[],'AR':[],'GrowingSpheres':[]}
cont_diver = {'FACE':[],'AR':[],'GrowingSpheres':[]}
cont_count_divers = {'FACE':[],'AR':[],'GrowingSpheres':[]}

for meth in methods:
    for i in counterfactuals[meth].index:
        original = pd.DataFrame(counterfactuals['factuals'].loc[i,:]).T
        CEs = pd.DataFrame(counterfactuals[meth].loc[i,:]).T

        df_orig = ce_helpers.visualise_changes(clf, d, encoder, method='CARLA',
                                               factual=original, CEs=CEs, scaler=scaler, only_changes=False)

        df_performance_1 = ce_helpers.evaluation(df_orig, d)

        validity[meth].append(df_performance_1.validity.item())
        cat_prox[meth].append(df_performance_1.cat_prox.item())
        cont_prox[meth].append(df_performance_1.cont_prox.item())
        sparsity[meth].append(df_performance_1.sparsity.item())
        cat_diver[meth].append(df_performance_1.cat_diver.item())
        cont_diver[meth].append(df_performance_1.cont_diver.item())
        cont_count_divers[meth].append(df_performance_1.cont_count_divers.item())

# remove None values
for key in validity.keys():
    validity[key] = [i for i in validity[key] if i is not None]
    cat_prox[key] = [i for i in cat_prox[key] if i is not None]
    cont_prox[key] = [i for i in cont_prox[key] if i is not None]
    sparsity[key] = [i for i in sparsity[key] if i is not None]
    cat_diver[key] = [i for i in cat_diver[key] if i is not None]
    cont_diver[key] = [i for i in cont_diver[key] if i is not None]
    cont_count_divers[key] = [i for i in cont_count_divers[key] if i is not None]


'''
---------SAVE---------
'''

if not os.path.exists(results_path):
    os.makedirs(results_path)

fnamefull = results_path + dataset.__name__ + '.txt'
with open(fnamefull, 'a') as f:
    print('%s \n' % dataset.__name__, file=f)
    print(' \t Validity  \t Cat_prox \t Cont_prox \t Sparsity \t Cat_diver \t Cont_diver \t Cont_count_divers \n', file=f)

    for method in validity.keys():
        txt = '{0}: \t {1:.2f} ({2:.2f}) \t {3:.2f} ({4:.2f}) \t {5:.2f} ({6:.2f}) \t {7:.2f} ({8:.2f}) \t {9:.2f} ({10:.2f}) \t {11:.2f} ({12:.2f}) \t {13:.2f} ({14:.2f}) '.format(method,
                                                                                                                                                                                     np.mean(validity[method]),
                                                                                                                                                                                     np.std(validity[method], ddof=1)/np.sqrt(np.size(validity[method])),
                                                                                                                                                                                     np.mean(cat_prox[method]),
                                                                                                                                                                                     np.std(cat_prox[method], ddof=1)/np.sqrt(np.size(cat_prox[method])),
                                                                                                                                                                                     np.mean(cont_prox[method]),
                                                                                                                                                                                     np.std(cont_prox[method], ddof=1)/np.sqrt(np.size(cont_prox[method])),
                                                                                                                                                                                     np.mean(sparsity[method]),
                                                                                                                                                                                     np.std(sparsity[method], ddof=1)/np.sqrt(np.size(sparsity[method])),
                                                                                                                                                                                     np.mean(cat_diver[method]),
                                                                                                                                                                                     np.std(cat_diver[method], ddof=1)/np.sqrt(np.size(cat_diver[method])),
                                                                                                                                                                                     np.mean(cont_diver[method]),
                                                                                                                                                                                     np.std(cont_diver[method], ddof=1)/np.sqrt(np.size(cont_diver[method])),
                                                                                                                                                                                     np.mean(cont_count_divers[method]),
                                                                                                                                                                                     np.std(cont_count_divers[method], ddof=1)/np.sqrt(np.size(cont_count_divers[method]))
                                                                                                                                                                                     )
        print(txt, file=f)

    print('\n -----\n', file=f)