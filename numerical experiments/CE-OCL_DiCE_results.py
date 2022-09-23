import Datasets as DS
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import ce_helpers
from itertools import chain
import dice_ml
import os
import pandas as pd


'''
You need to adjust the directory to the data folder and the results.
You need to change the dataset. 
You need to change the model (alg).

Use trust region or not? To compare with other carla method, we do not use the TR!  
'''
wd = 'data/'
dataset = DS.give_me_some_credit
act = DS.adult_actionability
alg = 'gbm'
# DiCE_methods = ['random', 'genetic', 'kdtree']
DiCE_methods = []
# results_path = '/Users/tabearober/Documents/Counterfactuals/CE-OCL/results_carla/'
results_path = 'results/'
# tr_region = False
# counterfactuals = 1
# actionability = False
tr_region = True
counterfactuals = 3


if tr_region:
    tr = 'with trust region'
else:
    tr = 'without trust region'


# extra actionability constraints can be included
# for example age can only get larger, or some variables are integer, etc.
# for now only available for the adult dataset
actionability = False
act = DS.adult_actionability

if actionability:
    a = 'with extra actionability constraints'
else:
    a = 'without extra actionability constraints'

'''
---------DATA---------
'''
df, df_train, df_test, d, clf, df_factuals, model_master, encoder, scaler = dataset(wd, alg)


'''
---------PREP for CE-OCL and DiCE---------
'''
target = d['target']
numerical = d['numerical']
categorical = d['categorical']

mapping = {}
for i in range(len(categorical)):
    mapping[categorical[i]] = [categorical[i] + '_' + s for s in list(encoder.categories_[i])]

# immutable features
I = [mapping.get(key) for key in list(set(d['immutable']).intersection(categorical))]
I = list(set(list(chain.from_iterable(I))).intersection(df.columns))
I.extend(set(d['immutable']).intersection(numerical))
# categorical features one-hot encoded
F_b = df.columns.difference(numerical + [target])


'''
---------PREP for CE-OCL---------
'''
# X: complete dataset without target
X = df.drop(target,axis=1)
# X1: point in X that have 1 as label. They will be used as trust region.
y_ix_1 = np.where(df[target]==1)
if dataset.__name__ == 'give_me_some_credit':
    X1 = X.iloc[y_ix_1[0],:].copy().reset_index(drop=True, inplace=False).iloc[:20000,:]
else:
    X1 = X.iloc[y_ix_1[0], :].copy().reset_index(drop=True, inplace=False)
# X1 = X1.head(10000)

sp = True
mu = 10000
# tr_region = True
enlarge_tr = False

# features that can only increase (become larger)
L = []
# conditionally mutable features, such as education level (can only take on higher categories)
Pers_I = []
# features that can only be positive
P = []
# dictionary with one-hot encoding used to ensure coherence
F_coh = {}
# integer features
F_int = []
# F_r
F_r = numerical


'''
---------PREP for DiCE---------
'''
features = df_train.drop(target,axis=1).columns
features_to_vary = [ele for ele in features if ele not in I]

# Step 1: dice_ml.Data
data = dice_ml.Data(dataframe=df_train, continuous_features=numerical, outcome_name=target)

# Step 2: dice_ml.Model
m = dice_ml.Model(model=clf, backend="sklearn")

# Step 3: dice_ml.Dice
# exp = dice_ml.Dice(data, m, method=meth)


'''
---------RUN---------
'''

validity = {'CE-OCL':[],'DiCE_random':[],'DiCE_genetic':[],'DiCE_kdtree':[]}
cat_prox = {'CE-OCL':[],'DiCE_random':[],'DiCE_genetic':[],'DiCE_kdtree':[]}
cont_prox = {'CE-OCL':[],'DiCE_random':[],'DiCE_genetic':[],'DiCE_kdtree':[]}
sparsity = {'CE-OCL':[],'DiCE_random':[],'DiCE_genetic':[],'DiCE_kdtree':[]}
cat_diver = {'CE-OCL':[],'DiCE_random':[],'DiCE_genetic':[],'DiCE_kdtree':[]}
cont_diver = {'CE-OCL':[],'DiCE_random':[],'DiCE_genetic':[],'DiCE_kdtree':[]}
cont_count_divers = {'CE-OCL':[],'DiCE_random':[],'DiCE_genetic':[],'DiCE_kdtree':[]}


for u_index in range(len(df_factuals)):
# for u_index in range(3):
    print('u_index: %i' % u_index)
    u = df_factuals.drop(target,axis=1).iloc[u_index,:]

    if actionability:
        L, Pers_I, P, F_coh, F_int, F_r, I, feature_ranges = act(df, d, encoder, u)

        ## CE-OCL
        df_performance_1 = pd.DataFrame()

        enlarge_tr = False
        try:
            CEs, CEs_, final_model = ce_helpers.opt(X, X1, u, numerical, F_b, F_int, F_coh, I, L, Pers_I, P, sp, mu,
                                                    tr_region, enlarge_tr, counterfactuals, model_master, scaler)
        except:
            print('Trust region is being enlarged!')
            enlarge_tr = True
            CEs, CEs_, final_model = ce_helpers.opt(X, X1, u, numerical, F_b, F_int, F_coh, I, L, Pers_I, P, sp, mu,
                                                    tr_region, enlarge_tr, counterfactuals, model_master, scaler)

        df_orig = ce_helpers.visualise_changes(clf, d, encoder, method='CE-OCL', CEs=CEs, CEs_=CEs_)
        if len(df_orig) > 1:  # i.e. if there are more than just the factual instance
            df_performance_1 = ce_helpers.evaluation(df_orig, d)

            validity['CE-OCL'].append(df_performance_1.validity.item())
            cat_prox['CE-OCL'].append(df_performance_1.cat_prox.item())
            cont_prox['CE-OCL'].append(df_performance_1.cont_prox.item())
            sparsity['CE-OCL'].append(df_performance_1.sparsity.item())
            cat_diver['CE-OCL'].append(df_performance_1.cat_diver.item())
            cont_diver['CE-OCL'].append(df_performance_1.cont_diver.item())
            cont_count_divers['CE-OCL'].append(df_performance_1.cont_count_divers.item())
        else:
            print('------NO SOLUTION FOUND WITH CE-OCL------')

        # DiCE
        for meth in DiCE_methods:

            df_performance_1 = pd.DataFrame()
            print('-----DiCE_%s-----' % meth)

            exp = dice_ml.Dice(data, m, method=meth)

            if meth == 'random':
                e1 = exp.generate_counterfactuals(pd.DataFrame(u).T, total_CFs=counterfactuals,
                                                  desired_class="opposite", random_seed=0,
                                                  features_to_vary=features_to_vary, permitted_ranges = feature_ranges)
            else:
                # e1 = exp.generate_counterfactuals(pd.DataFrame(u).T, total_CFs=3, desired_class="opposite",
                #                                   features_to_vary = features_to_vary)
                try:
                    e1 = exp.generate_counterfactuals(pd.DataFrame(u).T, total_CFs=counterfactuals,
                                                      desired_class="opposite",
                                                      features_to_vary=features_to_vary, permitted_range = feature_ranges)
                except:
                    continue

            df_orig = ce_helpers.visualise_changes(clf, d, encoder, method='DiCE', exp=e1, factual=pd.DataFrame(u).T,
                                                   scaler=scaler, only_changes=False)
            df_performance_1 = ce_helpers.evaluation(df_orig, d)

            validity['DiCE_%s' % meth].append(df_performance_1.validity.item())
            cat_prox['DiCE_%s' % meth].append(df_performance_1.cat_prox.item())
            cont_prox['DiCE_%s' % meth].append(df_performance_1.cont_prox.item())
            sparsity['DiCE_%s' % meth].append(df_performance_1.sparsity.item())
            cat_diver['DiCE_%s' % meth].append(df_performance_1.cat_diver.item())
            cont_diver['DiCE_%s' % meth].append(df_performance_1.cont_diver.item())
            cont_count_divers['DiCE_%s' % meth].append(df_performance_1.cont_count_divers.item())

    else:
        ## CE-OCL
        df_performance_1 = pd.DataFrame()

        enlarge_tr = False
        try:
            CEs, CEs_, final_model = ce_helpers.opt(X, X1, u, numerical, F_b, F_int, F_coh, I, L, Pers_I, P, sp, mu,
                               tr_region, enlarge_tr, counterfactuals, model_master, scaler)
        except:
            print('Trust region is being enlarged!')
            enlarge_tr = True
            CEs, CEs_, final_model = ce_helpers.opt(X, X1, u, numerical, F_b, F_int, F_coh, I, L, Pers_I, P, sp, mu,
                                                    tr_region, enlarge_tr, counterfactuals, model_master, scaler)


        df_orig = ce_helpers.visualise_changes(clf, d, encoder, method = 'CE-OCL', CEs=CEs, CEs_ = CEs_)
        if len(df_orig) > 1: # i.e. if there are more than just the factual instance
            df_performance_1 = ce_helpers.evaluation(df_orig, d)

            validity['CE-OCL'].append(df_performance_1.validity.item())
            cat_prox['CE-OCL'].append(df_performance_1.cat_prox.item())
            cont_prox['CE-OCL'].append(df_performance_1.cont_prox.item())
            sparsity['CE-OCL'].append(df_performance_1.sparsity.item())
            cat_diver['CE-OCL'].append(df_performance_1.cat_diver.item())
            cont_diver['CE-OCL'].append(df_performance_1.cont_diver.item())
            cont_count_divers['CE-OCL'].append(df_performance_1.cont_count_divers.item())
        else:
            print('------NO SOLUTION FOUND WITH CE-OCL------')

        # DiCE
        for meth in DiCE_methods:

            df_performance_1 = pd.DataFrame()
            print('-----DiCE_%s-----' % meth)

            exp = dice_ml.Dice(data, m, method=meth)

            if meth == 'random':
                e1 = exp.generate_counterfactuals(pd.DataFrame(u).T, total_CFs=counterfactuals, desired_class="opposite", random_seed=0,
                                            features_to_vary = features_to_vary)
            else:
                # e1 = exp.generate_counterfactuals(pd.DataFrame(u).T, total_CFs=3, desired_class="opposite",
                #                                   features_to_vary = features_to_vary)
                try: e1 = exp.generate_counterfactuals(pd.DataFrame(u).T, total_CFs=counterfactuals, desired_class="opposite",
                                            features_to_vary = features_to_vary)
                except:
                    continue

            df_orig = ce_helpers.visualise_changes(clf, d, encoder, method='DiCE', exp = e1, factual=pd.DataFrame(u).T, scaler=scaler, only_changes=False)
            df_performance_1 = ce_helpers.evaluation(df_orig, d)

            validity['DiCE_%s' % meth].append(df_performance_1.validity.item())
            cat_prox['DiCE_%s' % meth].append(df_performance_1.cat_prox.item())
            cont_prox['DiCE_%s' % meth].append(df_performance_1.cont_prox.item())
            sparsity['DiCE_%s' % meth].append(df_performance_1.sparsity.item())
            cat_diver['DiCE_%s' % meth].append(df_performance_1.cat_diver.item())
            cont_diver['DiCE_%s' % meth].append(df_performance_1.cont_diver.item())
            cont_count_divers['DiCE_%s' % meth].append(df_performance_1.cont_count_divers.item())

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
---------SAVE RESULTS---------
'''

if not os.path.exists(results_path):
    os.makedirs(results_path)

fnamefull = results_path + dataset.__name__ + '.txt'
with open(fnamefull, 'a') as f:
    # print('--->', file=f)
    print('%s \n' % dataset.__name__, file=f)
    # print('------', file=f)
    print('%s -- %s -- %i counterfactual(s) -- %s \n' % (alg.upper(), tr, counterfactuals, a), file=f)
    # print('------', file=f)
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