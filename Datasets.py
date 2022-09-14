import pandas as pd
import pickle
import os
import opticl as oc
import ce_helpers


def adult(wd, alg = 'rf'):
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
    path = wd + 'adult/%s_adult.pkl' % alg.upper()
    if os.path.exists(path):
        with open(path, 'rb') as f:
            clf = pickle.load(f)
    else:
        print('Train model: %s' % alg)
        clf, perf = oc.run_model(df_train.drop(d['target'],axis=1),
                                 df_train[d['target']],
                                 df_test.drop(d['target'], axis=1),
                                 df_test[d['target']],
                                 alg,
                                 'adult',
                                 'binary',
                                 save_path = path)
        # filename = 'results/' + model_choice + '_' + outcome + '_trained.pkl'
        with open(path, 'wb') as f:
            print(f'saving... {path}')
            pickle.dump(clf, f)

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

    # if not os.path.exists('results_carla/'):
    #     os.makedirs('results_carla/')

    constraintL = oc.ConstraintLearning(X_train, y_train, clf, alg)
    constraint_add = constraintL.constraint_extrapolation('binary')
    constraint_add.to_csv(wd + 'adult/%s_%s_model.csv' % (alg, 'adult'), index=False)

    ## create model master
    m = {
        'outcome': 'counterfactual_adult',
        'model_type': alg,
        'save_path': '%sadult/%s_%s_model.csv' % (wd, alg, 'adult'),
        'task': 'binary',
        'objective': 0,
        'lb': 0.5,
        'ub': None,
        'SCM_counterfactuals': None,
        'features': [[col for col in df.columns.drop(d['target'])]]
    }

    model_master = pd.DataFrame(m)

    return df, df_train, df_test, d, clf, df_factuals, model_master, encoder, scaler





def heloc(wd, alg = 'rf'):
    ## load data
    df = pd.read_csv(wd+'heloc/dataset_carla_heloc.csv')
    df_train = pd.read_csv(wd + 'heloc/train_carla_heloc.csv')
    df_test = pd.read_csv(wd + 'heloc/test_carla_heloc.csv')

    ## load readme file
    try:
        fhand = open(wd + 'heloc/README.txt')
    except:
        print("File cannot be opened.")
        exit()

    d = dict()

    for line in fhand:
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
    path = wd + 'heloc/%s_heloc.pkl' % alg.upper()
    if os.path.exists(path):
        with open(path, 'rb') as f:
            clf = pickle.load(f)
    else:
        print('Train model: %s' % alg)
        clf, perf = oc.run_model(df_train.drop(d['target'],axis=1),
                                 df_train[d['target']],
                                 df_test.drop(d['target'], axis=1),
                                 df_test[d['target']],
                                 alg,
                                 'heloc',
                                 'binary',
                                 save_path = path)
        # filename = 'results/' + model_choice + '_' + outcome + '_trained.pkl'
        with open(path, 'wb') as f:
            print(f'saving... {path}')
            pickle.dump(clf, f)

    ## load factual instances
    df_factuals = pd.read_csv(wd + 'heloc/factuals_heloc.csv')

    ## load encoder
    path = wd + 'heloc/encoder_heloc'
    with open(path, 'rb') as f:
        encoder = pickle.load(f)

    ## load scaler
    path = wd + 'heloc/scaler_heloc'
    with open(path, 'rb') as f:
        scaler = pickle.load(f)


    ## Save model
    y_train = df_train[d['target']]
    X_train = df_train.drop(d['target'], axis=1)

    # if not os.path.exists('results_carla/'):
    #     os.makedirs('results_carla/')

    constraintL = oc.ConstraintLearning(X_train, y_train, clf, alg)
    constraint_add = constraintL.constraint_extrapolation('binary')
    constraint_add.to_csv(wd + 'heloc/%s_%s_model.csv' % (alg, 'heloc'), index=False)

    ## create model master
    m = {
        'outcome': 'counterfactual_heloc',
        'model_type': alg,
        'save_path': '%sheloc/%s_%s_model.csv' % (wd, alg, 'heloc'),
        'task': 'binary',
        'objective': 0,
        'lb': 0.5,
        'ub': None,
        'SCM_counterfactuals': None,
        'features': [[col for col in df.columns.drop(d['target'])]]
    }

    model_master = pd.DataFrame(m)

    return df, df_train, df_test, d, clf, df_factuals, model_master, encoder, scaler







def compas(wd, alg = 'rf'):
    ## load data
    df = pd.read_csv(wd+'compas/dataset_carla_compas.csv')
    df_train = pd.read_csv(wd + 'compas/train_carla_compas.csv')
    df_test = pd.read_csv(wd + 'compas/test_carla_compas.csv')

    ## load readme file
    try:
        fhand = open(wd + 'compas/README.txt')
    except:
        print("File cannot be opened.")
        exit()

    d = dict()

    for line in fhand:
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
    path = wd + 'compas/%s_compas.pkl' % alg.upper()
    if os.path.exists(path):
        with open(path, 'rb') as f:
            clf = pickle.load(f)
    else:
        print('Train model: %s' % alg)
        clf, perf = oc.run_model(df_train.drop(d['target'],axis=1),
                                 df_train[d['target']],
                                 df_test.drop(d['target'], axis=1),
                                 df_test[d['target']],
                                 alg,
                                 'compas',
                                 'binary',
                                 save_path = path)
        # filename = 'results/' + model_choice + '_' + outcome + '_trained.pkl'
        with open(path, 'wb') as f:
            print(f'saving... {path}')
            pickle.dump(clf, f)

    ## load factual instances
    df_factuals = pd.read_csv(wd + 'compas/factuals_compas.csv')

    ## load encoder
    path = wd + 'compas/encoder_compas'
    with open(path, 'rb') as f:
        encoder = pickle.load(f)

    ## load scaler
    path = wd + 'compas/scaler_compas'
    with open(path, 'rb') as f:
        scaler = pickle.load(f)


    ## Save model
    y_train = df_train[d['target']]
    X_train = df_train.drop(d['target'], axis=1)

    # if not os.path.exists('results_carla/'):
    #     os.makedirs('results_carla/')

    constraintL = oc.ConstraintLearning(X_train, y_train, clf, alg)
    constraint_add = constraintL.constraint_extrapolation('binary')
    constraint_add.to_csv(wd + 'compas/%s_%s_model.csv' % (alg, 'compas'), index=False)

    ## create model master
    m = {
        'outcome': 'counterfactual_compas',
        'model_type': alg,
        'save_path': '%scompas/%s_%s_model.csv' % (wd, alg, 'compas'),
        'task': 'binary',
        'objective': 0,
        'lb': 0.5,
        'ub': None,
        'SCM_counterfactuals': None,
        'features': [[col for col in df.columns.drop(d['target'])]]
    }

    model_master = pd.DataFrame(m)

    return df, df_train, df_test, d, clf, df_factuals, model_master, encoder, scaler






def give_me_some_credit(wd, alg = 'rf'):
    ## load data
    df = pd.read_csv(wd+'give_me_some_credit/dataset_carla_give_me_some_credit.csv')
    df_train = pd.read_csv(wd + 'give_me_some_credit/train_carla_give_me_some_credit.csv')
    df_test = pd.read_csv(wd + 'give_me_some_credit/test_carla_give_me_some_credit.csv')

    ## load readme file
    try:
        fhand = open(wd + 'give_me_some_credit/README.txt')
    except:
        print("File cannot be opened.")
        exit()

    d = dict()

    for line in fhand:
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
    path = wd + 'give_me_some_credit/%s_give_me_some_credit.pkl' % alg.upper()
    if os.path.exists(path):
        with open(path, 'rb') as f:
            clf = pickle.load(f)
    else:
        print('Train model: %s' % alg)
        clf, perf = oc.run_model(df_train.drop(d['target'],axis=1),
                                 df_train[d['target']],
                                 df_test.drop(d['target'], axis=1),
                                 df_test[d['target']],
                                 alg,
                                 'give_me_some_credit',
                                 'binary',
                                 save_path = path)
        # filename = 'results/' + model_choice + '_' + outcome + '_trained.pkl'
        with open(path, 'wb') as f:
            print(f'saving... {path}')
            pickle.dump(clf, f)

    ## load factual instances
    df_factuals = pd.read_csv(wd + 'give_me_some_credit/factuals_give_me_some_credit.csv')

    ## load encoder
    path = wd + 'give_me_some_credit/encoder_give_me_some_credit'
    with open(path, 'rb') as f:
        encoder = pickle.load(f)

    ## load scaler
    path = wd + 'give_me_some_credit/scaler_give_me_some_credit'
    with open(path, 'rb') as f:
        scaler = pickle.load(f)


    ## Save model
    y_train = df_train[d['target']]
    X_train = df_train.drop(d['target'], axis=1)

    # if not os.path.exists('results_carla/'):
    #     os.makedirs('results_carla/')

    constraintL = oc.ConstraintLearning(X_train, y_train, clf, alg)
    constraint_add = constraintL.constraint_extrapolation('binary')
    constraint_add.to_csv(wd + 'give_me_some_credit/%s_%s_model.csv' % (alg, 'give_me_some_credit'), index=False)

    ## create model master
    m = {
        'outcome': 'counterfactual_give_me_some_credit',
        'model_type': alg,
        'save_path': '%sgive_me_some_credit/%s_%s_model.csv' % (wd, alg, 'give_me_some_credit'),
        'task': 'binary',
        'objective': 0,
        'lb': 0.5,
        'ub': None,
        'SCM_counterfactuals': None,
        'features': [[col for col in df.columns.drop(d['target'])]]
    }

    model_master = pd.DataFrame(m)

    return df, df_train, df_test, d, clf, df_factuals, model_master, encoder, scaler






