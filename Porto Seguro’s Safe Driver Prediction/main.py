from argparse import ArgumentParser
import pandas as pd
import utils.preprocessing as pp
import utils.train as train
import time
from sklearn.linear_model import LogisticRegressionCV

def main(args):

    # 0. Load Data
    data_path = './dataset/'
    train_df = pd.read_csv(data_path + 'train.csv')
    test_df = pd.read_csv(data_path + 'test.csv')
    print("[Load Dataset] TRAIN {} / TEST {}".format(train_df.shape, test_df.shape))

    # 1-1. Data Encoding
    test_df['target'] = 0
    test_df['type'] = 'test'
    train_df['type'] = 'train'
    del train_df['id']
    test_id = test_df['id']

    all_df = pd.DataFrame()
    all_df = pd.concat([train_df, test_df], join='outer', axis=0)
    all_df = pp.preprocessing_for_catefory(all_df)

    train_df = all_df[all_df['type'] == 'train']
    test_df = all_df[all_df['type'] == 'test']
    del train_df['type']
    del test_df['type']
    print("[Load Dataset] TRAIN {} / TEST {}".format(train_df.shape, test_df.shape))

    train_df = train_df.head(100)

    # 1-2. Data Split X, y
    train_y = train_df['target']
    train_X = train_df.drop(columns='target', axis=1)
    test_X  = test_df.drop(columns='target', axis=1)


    # 2-1. Train - LightGBM
    params = {'objective': 'binary ', 'seed': 777, 'learning_rate': 0.14, 'is_unbalance': False,
              'drop_rate': 0.1, 'min_child_samples': 10, 'min_child_weight': 150, 'subsample': 0.85}
    lgbm = train.LightGBM(X=train_X, y=train_y, use_cv=True, k_fold=5, **params)
    lgbm_model = lgbm.get_model()
    params = {'objective': 'binary:logistic', 'seed': 777,
              'num_round': 100, 'early_stopping_rounds': 20, 'learning_rate': 0.14, 'max_depth': 4,
              'gamma': 0.05,  'reg_alpha': 0.01}
    xgb = train.XGB(X=train_X, y=train_y, use_cv=True, k_fold=5, **params)
    xgb_model = xgb.get_model()

    clfs = [lgbm_model, xgb_model]
    clf = LogisticRegressionCV(refit=False)
    stacker = train.Stacker(X=train_X, y=train_y, T=test_X, k_fold=2, stacker_1=clfs, stacker_2=clf)
    stacker.model_fit()
    exit()

    # 2-1. Train - LightGBM
    params = {'objective': 'binary ', 'seed': 777, 'learning_rate': 0.14, 'is_unbalance': False,
              'drop_rate': 0.1, 'min_child_samples': 10, 'min_child_weight': 150, 'subsample': 0.85}
    lgbm = train.LightGBM(X=train_X, y=train_y, use_cv=True, k_fold=5, **params)
    lgbm.model_fit_custom()
    lgbm.model_predict(train_X, train_y)

    # 3-1. Test
    lgbm.model_test(test_X, test_id)

    # 2-1. Train
    grid_dict = {
        'learning_rate': [0.1, 0.12, 0.14, 0.16],
        'max_depth': [3, 4, 5],
        'min_child_weight': [10, 50, 100, 150],
        'gamma': [i / 10.0 for i in range(0, 5)],
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
    }

    params = {'objective': 'binary:logistic', 'seed': 777,
              'num_round': 100, 'early_stopping_rounds': 20, 'learning_rate': 0.14, 'max_depth': 4,
              'gamma': 0.05,  'reg_alpha': 0.01}
    xgb = train.XGB(X=train_X, y=train_y, use_cv=True, k_fold=5, **params)
    xgb.model_fit_custom()
    xgb.model_predict(train_X, train_y)

    # 3-1. Test
    xgb.model_test(test_X, test_id)



    '''
    start_time = time.time()
    grid_param = {k:v}
    print("[Hyperparameter Tunning][START][{}] {} ".format(start_time, grid_param))
    xgb = train.XGB(X=train_X, y=train_y, use_cv=True, k_fold=3, **params)
    xgb.model_gridsearch_fit(grid_param)
    xgb.model_predict()
    params = xgb.get_params()
    print(params)
    print("[Hyperparameter Tunning][END] - {} sec ".format(time.time()-start_time))
    print("\n")
    '''

    # 3. Test
    _, y_prob = xgb.model_test()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--cv', type=int, default=5)
    args = parser.parse_args()
    main(args)


