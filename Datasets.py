import pandas as pd
import pickle
import os
import opticl as oc
import ce_helpers


def adult(wd):
    ## load data
    df = pd.read_csv(wd+'adult/dataset_carla_adult.csv')
    df_train = pd.read_csv(wd + 'adult/train_carla_adult.csv')
    df_test = pd.read_csv(wd + 'adult/test_carla_adult.csv')

    ## load readme file
    try:
        fhand = open(wd + 'adult/README.txt')
    except:
        print("File cannot be opened.")
        exit()

    d = dict()

    for line in fhand:
        # line = line.strip()
        # words = line.split()
        line = line.strip()
        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.replace(',', '')
        line = line.replace("'", '')
        words = line.split()
        if line.startswith("categorical:"):
            d['categorical'] = words[1:]

        elif line.startswith('continuous:'):
            d['numerical'] = words[1:]

        elif line.startswith('immutables:'):
            d['immutable'] = words[1:]

        elif line.startswith('target:'):
            d['target'] = words[1]

    ## load predictive model
    path = wd + 'adult/RF_adult.pkl'
    with open(path, 'rb') as f:
        clf = pickle.load(f)

    ## load factual instances
    df_factuals = pd.read_csv(wd + 'adult/factuals_adult.csv')

    ## load encoder
    path = wd + 'adult/encoder_adult'
    with open(path, 'rb') as f:
        encoder = pickle.load(f)

    ## load scaler
    path = wd + 'adult/scaler_adult'
    with open(path, 'rb') as f:
        scaler = pickle.load(f)


    ## Save model
    y_train = df_train[d['target']]
    X_train = df_train.drop(d['target'], axis=1)

    if not os.path.exists('results_carla/'):
        os.makedirs('results_carla/')

    constraintL = oc.ConstraintLearning(X_train, y_train, clf, 'rf')
    constraint_add = constraintL.constraint_extrapolation('binary')
    constraint_add.to_csv('results_carla/%s_%s_model.csv' % ('rf', 'adult'), index=False)

    ## create model master
    m = {
        'outcome': 'counterfactual_adult',
        'model_type': 'rf',
        'save_path': 'results_carla/%s_%s_model.csv' % ('rf', 'adult'),
        'task': 'binary',
        'objective': 0,
        'lb': 0.5,
        'ub': None,
        'SCM_counterfactuals': None,
        'features': [[col for col in df.columns.drop('income')]]
    }

    model_master = pd.DataFrame(m)

    return df, df_train, df_test, d, clf, df_factuals, model_master, encoder, scaler


def label(row, filter_col):
    if any(row[filter_col] == 1):
        return 0
    else: return 1

def recreate_orig(df, d, encoder):
    target = d['target']
    numerical = d['numerical']
    categorical = d['categorical']
    # categorical = df.columns.difference(numerical + [target])

    mapping = {}
    for i in range(len(categorical)):
        mapping[categorical[i]] = [categorical[i] + '_' + s for s in list(encoder.categories_[i])]
    # mapping

    categorical_columns = []
    for key in mapping.keys():
        for v in mapping[key]:
            categorical_columns.append(v)
    # categorical_columns

    for c in categorical_columns:
        if c in df:
            continue
        else:
            feature = c.split('_')[0]
            filter_col = [col for col in df if col.startswith(feature)]
            df[c] = df.apply(lambda row: label(row, filter_col), axis=1)

    df1 = pd.DataFrame()
    for v in numerical + [target]:
        df1[v] = df[v]

    for v in mapping:
        df1[v] = df[mapping[v]].apply(lambda row: ce_helpers.reverse_dummies(row, df, mapping, v), axis=1)

    for c in df1.columns.difference(numerical+[target]):
        df1[c] = df1.apply(lambda row: ce_helpers.value_names(row, c), axis=1)

    return df, df1