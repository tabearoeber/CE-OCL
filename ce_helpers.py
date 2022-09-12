import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from pyomo.environ import *
import opticl as oc
import os
import pickle
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import scipy
import ce_helpers
import embed_mip as em



######## PREPARATION ########


def prep_data(X, y, numerical, one_hot_encoding=True, scaling=True):
    '''
    X: features, not processed
    y: outcome variable
    numerical: list with numerical features
    '''

    categorical = X.columns.difference(numerical)
    F_r = numerical

    # train/test split
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X,
                                                                            y,
                                                                            test_size=0.2,
                                                                            random_state=0,
                                                                            stratify=y)  # this has to be changed

    # create transformer
    if scaling and one_hot_encoding:
        transformations = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
                          ('num', MinMaxScaler(), numerical)])
    elif scaling and not one_hot_encoding:
        transformations = ColumnTransformer(transformers=[('cat', 'passthrough', categorical),
                                                          ('num', MinMaxScaler(), numerical)])
    elif not scaling and one_hot_encoding:
        transformations = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
                                                          ('num', 'passthrough', numerical)])

    # create processing pipeline
    data_pip = Pipeline(steps=[('preprocessor', transformations)])

    # fit pipeline on training data only
    X_array = data_pip.fit_transform(X_train_temp)

    # get column names
    categorical_names = []
    if scaling and one_hot_encoding:
        categorical_names = list(data_pip['preprocessor'].transformers_[0][1].get_feature_names(categorical))
        column_names = categorical_names + numerical
    elif scaling:
        categorical_names = list(categorical)
        column_names = categorical_names + numerical
    else:
        categorical_names = list(data_pip['preprocessor'].transformers_[0][1].get_feature_names(categorical))
        column_names = categorical_names + numerical

    if data_pip['preprocessor'].sparse_output_:
        temp_array = scipy.sparse.csr_matrix(np.array(X_array).all())

        X_cat = pd.DataFrame(temp_array.toarray(), columns=categorical_names)
        X_ = X.drop(list(categorical), axis=1, inplace=False).join(X_cat)

        ## apply data_pip to X_train_temp and X_test_temp to create final X_train and X_test sets
        X_train = pd.DataFrame(scipy.sparse.csr_matrix(np.array(data_pip.transform(X_train_temp)).all()).toarray(),
                               columns=column_names)

        X_test = pd.DataFrame(scipy.sparse.csr_matrix(np.array(data_pip.transform(X_test_temp)).all()).toarray(),
                              columns=column_names)
        X_test = X_test.set_index(pd.Index(list(X_test_temp.index)))

        X = pd.DataFrame(data_pip.transform(X), columns=column_names)

    else:
        X_ = pd.DataFrame(X_array, columns=column_names)

        ## apply data_pip to X_train_temp and X_test_temp to create final X_train and X_test sets
        X_train = pd.DataFrame(data_pip.transform(X_train_temp), columns=column_names)
        X_test = pd.DataFrame(data_pip.transform(X_test_temp), columns=column_names)
        X_test = X_test.set_index(pd.Index(list(X_test_temp.index)))

        X = pd.DataFrame(data_pip.transform(X), columns=column_names)

    return X, X_train, X_test, y_train_temp, y_test_temp, categorical_names, data_pip


def train_models(outcome_dict, version):
    '''
    train the predictive models for each outcome.
    outcome_dict = dictionary containing for each outcome: the predictive models to train, the type of task,
    the features, the train and test data
    '''
    performance = pd.DataFrame()

    if not os.path.exists('results/'):
        os.makedirs('results/')

    for outcome in outcome_dict:
        print(f'Learning a constraint for {outcome}')
        task_type = outcome_dict[outcome]['task']
        alg_list = outcome_dict[outcome]['alg_list']
        X_train = outcome_dict[outcome]['X_train'][outcome_dict[outcome]['X features']]
        X_test = outcome_dict[outcome]['X_test'][outcome_dict[outcome]['X features']]
        y_train = outcome_dict[outcome]['y_train']
        y_test = outcome_dict[outcome]['y_test']

        for alg in alg_list:
            if not os.path.exists('results/%s/' % alg):
                os.makedirs('results/%s/' % alg)
            print(f'Training {alg}')
            s = 0

            alg_run = 'rf_shallow' if alg == 'rf' else alg

            m, perf = oc.run_model(X_train, y_train, X_test, y_test, alg_run, outcome, task=task_type,
                                   seed=s, cv_folds=5,
                                   save=False
                                   )

            ## Save model
            constraintL = oc.ConstraintLearning(X_train, y_train, m, alg)
            constraint_add = constraintL.constraint_extrapolation(task_type)
            constraint_add.to_csv('results/%s/%s_%s_model.csv' % (alg, version, outcome), index=False)

            ## Extract performance metrics
            try:
                perf['auc_train'] = roc_auc_score(y_train >= threshold, m.predict(X_train))
                perf['auc_test'] = roc_auc_score(y_test >= threshold, m.predict(X_test))
            except:
                perf['auc_train'] = np.nan
                perf['auc_test'] = np.nan

            perf['seed'] = s
            perf['outcome'] = outcome
            perf['alg'] = alg
            perf['save_path'] = 'results/%s/%s_%s_model.csv' % (alg, version, outcome)

            perf.to_csv('results/%s/%s_%s_performance.csv' % (alg, version, outcome), index=False)

            performance = performance.append(perf)
            print()
    print('Saving the performance...')
    performance.to_csv('results/%s_performance.csv' % version, index=False)
    print('Done!')


def perf_trained_models(version, outcome_dict):
    performance = pd.read_csv('results/%s_performance.csv' % version)
    performance['task'] = 'continuous'
    for outcome in outcome_dict.keys():
        performance.loc[performance['outcome']==outcome, 'task'] = outcome_dict[outcome]['task']
    performance = performance.dropna(axis='columns')
    return performance


def load_model(algorithms, outcome_dict, CE_outcome):
    models = {}

    for outcome in outcome_dict:
        algorithm = algorithms[outcome]
        X_train = outcome_dict[outcome]['X_train']
        X_test = outcome_dict[outcome]['X_test']
        y_train = outcome_dict[outcome]['y_train']
        y_test = outcome_dict[outcome]['y_test']

        # Load predictive model
        if algorithm == 'rf': algorithm = 'rf_shallow'
        path = 'results/' + algorithm + '_' + outcome + '_trained.pkl'
        with open(path, 'rb') as f:
            clf = pickle.load(f)

        # if we load the predictive model to ensure validity, we want to extract X_test_0
        # i.e. the instances that were predicted as 0. We can use these as factual instances
        if outcome == CE_outcome:
            y_pred = clf.predict(X_test)
            y_pred_0 = np.where(y_pred == 0)
            X_test_0 = X_test.iloc[y_pred_0[0], :].copy()
            X_test_0.head()

        models[outcome] = clf

    return y_pred, y_pred_0, X_test_0, models


######## OPTIMIZATION ########
def opt(X, X1, u, F_r, F_b, F_int, F_coh, I, L, Pers_I, P, sp, mu, tr_region, enlarge_tr, num_counterfactuals,
        model_master, scaler=None, obj='l2'):
    sparsity_RHS = len(X.columns)

    conceptual_model = ce_helpers.CounterfactualExplanation(X, u, F_r, F_b, F_int, F_coh, mu, sparsity_RHS, I, L,
                                                            Pers_I, P, obj=obj, sparsity=sp, tr=tr_region)
    MIP_final_model = em.optimization_MIP(conceptual_model, conceptual_model.x, model_master, X1, tr=tr_region,
                                          enlarge_tr=enlarge_tr)
    opt = SolverFactory('gurobi_persistent')
    opt.set_instance(MIP_final_model)
    opt.set_gurobi_param('PoolSolutions', num_counterfactuals + 100)
    # opt.set_gurobi_param('PoolSolutions', num_counterfactuals)
    opt.set_gurobi_param('PoolSearchMode', 1)

    # opt.options['Solfiles'] = 'solution'+version
    results = opt.solve(MIP_final_model, load_solutions=True, tee=False)
    print('OBJ:', value(MIP_final_model.OBJ))

    solution = []
    for i in X.columns:
        solution.append(value(MIP_final_model.x[i]))
    print(f'The optimal solution is: {solution}')

    number_of_solutions = opt.get_model_attr('SolCount')
    # print(f"########################Num of solutions###################### {number_of_solutions}")
    CEs = pd.DataFrame([u])
    for i in range(number_of_solutions):
        opt.set_gurobi_param('SolutionNumber', i)
        suboptimal_solutions = opt.get_model_attr('Xn')

        vars_name_x = [opt.get_var_attr(MIP_final_model.x[i], 'VarName') for i in X.columns]
        vars_name_ix = [int(vars_name_x[i].replace('x', '')) for i in range(len(vars_name_x))]
        # print(vars_name_ix)
        vars_val_x = [suboptimal_solutions[i - 1] for i in vars_name_ix]
        solution_i = {X.columns[i]: vars_val_x[i] for i in range(len(vars_val_x))}
        solution_i = pd.DataFrame(solution_i, index=[0])
        CEs = CEs.append(solution_i)

    CEs.reset_index(drop=True, inplace=True)

    CEs['scaled_distance'] = [np.round(sum(abs(u[i] - CEs.loc[j, i]) for i in X.columns), 4) for j in CEs.index]
    # CEs['sparsity'] = [sum(1 if np.round(u[i]-CEs.loc[j,i], 3) != 0 else 0 for i in X.columns) for j in CEs.index]
    # CEs['obj value'] = [CEs.loc[j,'sparsity']*mu + CEs.loc[j,'scaled_distance'] for j in CEs.index]
    CEs = CEs.round(4).drop_duplicates()

    ix_names = ['original'] + ['sol' + str(i) for i in range(len(CEs.index))]
    ix = {i: ix_names[i] for i in range(len(CEs.index))}
    CEs = CEs.reset_index(drop=True).rename(index=ix)
    CEs = CEs.iloc[:num_counterfactuals + 1, :]

    # reverse scaling
    CEs_ = CEs.iloc[:, :-1].copy()

    if scaler is not None:
        try: scaled_xdata_inv = scaler['preprocessor'].named_transformers_['num'].inverse_transform(CEs_[F_r])
        except: scaled_xdata_inv = scaler.inverse_transform(CEs_[F_r])
        #        scaled_xdata_inv = scaler.inverse_transform(CEs_[F_r])
        CEs_.loc[:, F_r] = scaled_xdata_inv

    return CEs, CEs_, MIP_final_model



######## MAP BACK TO INPUT SPACE ########
def reverse_dummies(row, CEs_, F_coh, v):
    if sum(row[F_coh[v]]) > 1:
        return 'TWO DUMMIES 1'
    elif sum(row[F_coh[v]]) == 0:
        return 'NO DUMMY 1'
    else:
        for c in F_coh[v]:
            if row[c] == 1:
                return c


def ce_change(row, df, orig, c):
    if row[c] == orig[c][0]:
        return '-'
    else:
        return row[c]


def value_names(row, c):
    if row[c] == '-':
        return row[c]
    elif row[c] == 'TWO DUMMIES 1':
        return row[c]
    elif row[c] == 'NO DUMMY 1':
        return row[c]
    else:
        return row[c].split('_')[-1]


def vis_dataframe(dataset, CEs_, F_r, F_coh, target, only_changes=True):
    df = pd.DataFrame()
    for v in F_r:
        df[v] = CEs_[v]

    for v in F_coh:
        df[v] = CEs_[F_coh[v]].apply(lambda row: reverse_dummies(row, CEs_, F_coh, v), axis=1)

    df = df.round(2)
    df

    if only_changes:
        orig = df[:1]
        df = df[1:].copy()
        df1 = pd.DataFrame()
        for c in df.columns:
            df1[c] = df.apply(lambda row: ce_change(row, df1, orig, c), axis=1)

        df = pd.concat([orig, df1])

    for c in df.columns.difference(F_r):
        df[c] = df.apply(lambda row: value_names(row, c), axis=1)

    return df


def visualise_changes(df):
    orig = df[:1]
    df = df[1:].copy()
    df1 = pd.DataFrame()
    for c in df.columns:
        df1[c] = df.apply(lambda row: ce_change(row, df1, orig, c), axis=1)

    df = pd.concat([orig, df1])
    return df


######## EVALUATION ########


def prox_score(F_cat, F_r, CEs, number_of_solutions):
    cont_prox = -sum(
        sum(np.abs(CEs.loc['original', i] - CEs.loc[j, i]) for i in F_r) for j in CEs.index[1:]) / number_of_solutions
    if len(F_cat) > 0:
        cat_prox = 1 - sum(sum((CEs.loc['original', F_cat]) != (CEs.loc[i, F_cat])) for i in CEs.index[1:]) / (
                number_of_solutions * len(F_cat))
    else:
        cat_prox = None

    return cont_prox, cat_prox


def sparsity_score(CEs, number_of_solutions):
    orig = CEs[:1]
    sparsity = 1 - sum(
        sum(1 if orig.squeeze()[i] != CEs.loc[j, i] else 0 for i in CEs.columns) for j in CEs.index[1:]) / (
                       number_of_solutions * len(CEs.columns))
    return sparsity


def diversity_score(CEs, F_cat, F_r, number_of_solutions):
    # continuous variables
    cont_diver = None
    if number_of_solutions > 1:
        cont_diver_numerator = 0
        for jx, j in enumerate(CEs.index[1:-1]):
            for i in CEs.index[jx + 1:]:
                if i != j:
                    cont_diver_numerator += sum(
                        np.abs(np.round(CEs.loc[i, F_r].astype(float), 4) - np.round(CEs.loc[j, F_r].astype(float), 4)))
        cont_diver_denominator = number_of_solutions * (number_of_solutions - 1) / 2
        cont_diver = cont_diver_numerator / cont_diver_denominator

    # categorical variables
    cat_diver = None
    if number_of_solutions > 1:
        if len(F_cat) > 0:
            cat_diver_numerator = 0
            for jx, j in enumerate(CEs.index[1:-1]):
                for i in CEs.index[jx + 1:]:
                    if i != j:
                        cat_diver_numerator += sum(CEs.loc[i, F_cat] != CEs.loc[j, F_cat])
            cat_diver_denominator = number_of_solutions * (number_of_solutions - 1) / 2 * len(F_cat)
            cat_diver = cat_diver_numerator / cat_diver_denominator

    return cont_diver, cat_diver


def spars_divers_score(CEs, number_of_solutions):
    cont_count_divers = None
    if number_of_solutions > 1:
        sparsity_diver_numerator = 0
        for jx, j in enumerate(CEs.index[1:-1]):
            for i in CEs.index[jx + 1:]:
                if i != j:
                    sparsity_diver_numerator += sum(np.abs(CEs.loc[i, :] != CEs.loc[j, :]))
        sparsity_diver_denominator = number_of_solutions * (number_of_solutions - 1) / 2 * len(CEs.columns)
        cont_count_divers = sparsity_diver_numerator / sparsity_diver_denominator

    return cont_count_divers


def evaluation(model, CEs, numerical, categorical, rounding=True, CEs_=None):
    number_of_solutions = len(CEs.index) - 1
    '''
    model = predictive model
    CEs = dataframe with factual instance in the top row and all counterfactuals in the other rows, already reverse scaled!
    numerical = set of real features
    categorical = set of categorical features (not one-hot encoded)
    CEs_ = like CEs, but scaled
    '''
    if CEs_ is None:
        # then we evaluate DiCE results, where we do not have CEs_
        validity = sum(model.predict(CEs.iloc[1:, :])) / number_of_solutions
    else:
        try: validity = sum(model.predict(CEs_.drop('scaled_distance',axis=1).iloc[1:, :])) / number_of_solutions
        except: validity = 1

    cont_prox, cat_prox = prox_score(categorical, numerical, CEs, number_of_solutions)

    sparsity = sparsity_score(CEs, number_of_solutions)

    cont_diver, cat_diver = diversity_score(CEs, categorical, numerical, number_of_solutions)

    cont_count_divers = spars_divers_score(CEs, number_of_solutions)

    CE_perf = pd.DataFrame([[validity, cat_prox, cont_prox, sparsity, cat_diver, cont_diver, cont_count_divers]],
                           columns=['validity', 'cat_prox', 'cont_prox', 'sparsity', 'cat_diver', 'cont_diver',
                                    'cont_count_divers'])

    if rounding == True:
        CE_perf = CE_perf.round(2)

    return CE_perf


def CounterfactualExplanation(X, u, F_r, F_b, F_int, F_coh, mu, sparsity_RHS, I, L, Pers_I, P, obj='l2', sparsity=True,
                              tr=False):
    '''
    u: given data point
    F_r = set of real features that describe u
    F_b = set of binary features that describe u
    F_i = set of integer features that describe u
    I = set of immutable features
    Pers_I = variables that are conditionally mutable
    L = set of features that should only increase
    P = set of features that must take on a positive value
    MA = set of mutable and actionable features
    MU = set of mutable but unactionable features
    mu = positive hyperparameter per the sparsity constraint
    obj = which objective function to use?
    sparsity = True or False. include sparsity constraint or not?
    '''
    # complete set of features
    F = X.columns
    # print(F)
    big_M = 10000

    model = ConcreteModel('CE')

    'Decision variables'
    model.z = Var(F, domain=Binary,
                  name=['Zaux_%s' % str(ce) for ce in F])  # auxiliary vars for the sparsity constraint
    model.t = Var(F, domain=PositiveReals,
                  name=['Taux_%s' % str(ce) for ce in F])  # auxiliary vars for the MAD objective function
    model.x = Var(F, domain=Reals, name=['ce_%s' % str(ce) for ce in F])  # counterfactual features
    model.e = Var(F, domain=Reals, name=['epsilon_%s' % str(x) for x in F])

    for i in F_b:
        model.x[i].domain = Binary

    for i in F_int:
        model.x[i].domain = NonNegativeIntegers

    for cat in F_coh.keys():
        model.add_component('coherence_' + cat, Constraint(expr=sum(model.x[i] for i in F_coh[cat]) == 1))

    'Objective function'

    def obj_function_l2norm(model, sparsity=sparsity):
        return sum((u[i] - model.x[i]) ** 2 for i in F) + mu * sum(model.z[i] for i in F)

    def obj_function_MAD(model):
        MAD = {
            i: stats.median_absolute_deviation(X[i]) if stats.median_absolute_deviation(X[i]) > 1e-4 else 1.48 * np.std(
                X[i]) for i in F}
        return sum(model.t[i] / MAD[i] for i in F_r + F_int) + sum(model.t[i] / MAD[i] for i in F_b) + mu * sum(
            model.z[i] for i in F)

    def obj_function_l1norm(model):
        return sum(model.t[i] for i in F) + mu * sum(model.z[i] for i in F)

    assert obj in ['l2', 'l1MAD'], "Invalid objective function; please choose between l2, l1, l1MAD"
    if obj == 'l2':
        model.OBJ = Objective(rule=obj_function_l2norm, sense=minimize)
    elif obj == 'l1MAD':
        model.OBJ = Objective(rule=obj_function_MAD, sense=minimize)
    elif obj == 'l1':
        model.OBJ = Objective(rule=obj_function_l1norm, sense=minimize)

        'Auxiliary constraint for t'

        def MAD1(model, i):
            return model.t[i] >= u[i] - model.x[i]

        def MAD2(model, i):
            return model.t[i] >= - u[i] + model.x[i]

        model.Constraint01 = Constraint(F, rule=MAD1)
        model.Constraint02 = Constraint(F, rule=MAD2)

    if not tr:
        def constraint_CTR2(model, i):
            return model.e[i] == 0

        model.ConstraintClusteredTrustRegion2 = Constraint(F, rule=constraint_CTR2)

    if sparsity == True:
        'Sparsity constraints'

        def sparsity11(model, i):
            return model.x[i] - u[i] <= big_M * model.z[i]

        model.Constraint11 = Constraint(F, rule=sparsity11)

        def sparsity12(model, i):
            return -(model.x[i] - u[i]) <= big_M * model.z[i]

        model.Constraint12 = Constraint(F, rule=sparsity12)

    # This is not necessary since we already have a penalizing term in the objective function
    #     def sparsity2(model):
    #         return sum(model.z[i] for i in F) <= sparsity_RHS
    #     model.Constraint2 = Constraint(rule=sparsity2)

    'Immutable features constraints'

    for k in Pers_I:
        for j in k:
            # print(j)
            if j in u.index:
                if u[j] == 1:
                    I = I + k[:k.index(j)]
            else:
                I = I + k[:k.index(j)]

    def immutability(model, i):
        # print()
        return model.x[i] == u[i]

    model.Constraint3 = Constraint(I, rule=immutability)

    'Larger or equal to u[i]'

    def larger(model, i):
        return model.x[i] >= u[i]

    model.Constraint4 = Constraint(L, rule=larger)

    def positive(model, i):
        return model.x[i] >= 0

    model.Constraint5 = Constraint(P, rule=positive)

    return model




######## Only used for DiCE ########

def get_features_range(data_df, continuous_feature_names, categorical_feature_names, permitted_range_input=None):
    ranges = {}
    # Getting default ranges based on the dataset
    for feature_name in continuous_feature_names:
        ranges[feature_name] = [
            data_df[feature_name].min(), data_df[feature_name].max()]
    for feature_name in categorical_feature_names:
        ranges[feature_name] = data_df[feature_name].unique().tolist()
    feature_ranges_orig = ranges.copy()
    # Overwriting the ranges for a feature if input provided
    if permitted_range_input is not None:
        for feature_name, feature_range in permitted_range_input.items():
            ranges[feature_name] = feature_range
    return ranges, feature_ranges_orig


