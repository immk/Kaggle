import pandas as pd
import numpy as np
import xgboost
import lightgbm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer

def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_score():
    return make_scorer(gini_normalized, greater_is_better=True, needs_proba=True)

class LightGBM:

    def __init__(self, X, y, use_cv, k_fold, **kwargs):
        print("INIT {}".format(kwargs))
        self.X = X
        self.y = y
        self.use_cv = use_cv
        self.k_fold = k_fold
        self.lgbm = lightgbm.LGBMClassifier(**kwargs)

    def model_fit(self):
        if self.use_cv:
            params = self.lgbm.get_params()
            print(params)

            dtrain = lightgbm.Dataset(self.X, label=self.y)
            lgbm_cv = lightgbm.cv(params, dtrain, nfold=self.k_fold, metrics='auc', early_stopping_rounds=20)
            print(lgbm_cv)

        print("Model Fit")
        self.lgbm.fit(self.X, self.y, eval_metric=gini_score)

    def model_fit_custom(self):
        kfold = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=777).split(self.X, self.y)
        best_trees = []

        for idx, (train, val) in enumerate(kfold):
            print("[MODEL_LightGBM][FIT] Cross Validation {}/{}".format(idx+1, self.k_fold))
            X_train = self.X.iloc[train, :]
            y_train = self.y.iloc[train]
            X_val = self.X.iloc[val, :]
            y_val = self.y.iloc[val]

            self.lgbm.fit(X_train, y_train, eval_metric=gini_score)
            self.model_predict(X_val, y_val)
            best_trees.append(self.lgbm.best_iteration_)

        print("Model Fit")
        self.lgbm.fit(self.X, self.y, eval_metric=gini_score)

    def model_gridsearch_fit(self, grid):
        params = self.lgbm.get_params()
        lgbm_grid = GridSearchCV(estimator=self.lgbm, param_grid=grid, scoring=gini_score, n_jobs=-1, cv=self.k_fold, iid=False)
        lgbm_grid.fit(self.X, self.y)
        print(" >> Grid Scores\n{}".format(lgbm_grid.grid_scores_))
        print(" >> Best Params  {}".format(lgbm_grid.best_params_))
        print(" >> Best Score   {}".format(lgbm_grid.best_score_))

        self.lgbm = lgbm_grid.estimator
        params = self.lgbm.get_params()
        for k, v in lgbm_grid.best_params_.items():
            params[k] = v
        self.lgbm.set_params(params)
        self.model_fit()

    def model_predict(self, X, y):
        y_pred = self.lgbm.predict(X)
        y_prob = self.lgbm.predict_proba(X)[:, 1]
        print(gini_normalized(y, y_prob))

    def model_test(self, T, idx, file_path='./dataset/submission.csv'):
        y_pred = self.lgbm.predict(T)
        y_prob = self.lgbm.predict_proba(T)[:, 1]
        pd.DataFrame({'target': y_prob, 'id': idx}).to_csv(file_path, index=False)
        print("[TEST][SAVE Result] {}".format(file_path))
        return y_pred, y_prob

    def get_params(self):
        return self.lgbm.get_xgb_params()

    def get_model(self):
        return self.lgbm


class XGB:

    def __init__(self, X, y, use_cv, k_fold, **kwargs):
        print("INIT {}".format(kwargs))
        self.X = X
        self.y = y
        self.use_cv = use_cv
        self.k_fold = k_fold
        self.xgb = xgboost.XGBClassifier(**kwargs)

    def model_fit(self):

        if self.use_cv:
            params = self.xgb.get_xgb_params()
            dtrain = xgboost.DMatrix(self.X, label=self.y)
            xgb_cv = xgboost.cv(params, dtrain, num_boost_round=self.xgb.n_estimators, nfold=self.k_fold, metrics='auc', early_stopping_rounds=20)
            print(xgb_cv)
            self.xgb.set_params(xgb_cv.shape[0])

        print("Model Fit")
        self.xgb.fit(self.X, self.y, eval_metric='auc')

    def model_fit_custom(self):
        kfold = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=777).split(self.X, self.y)
        best_trees = []

        for idx, (train, val) in enumerate(kfold):
            print("[MODEL_XGB][FIT] Cross Validation {}/{}".format(idx+1, self.k_fold))
            X_train = self.X.iloc[train, :]
            y_train = self.y.iloc[train]
            X_val = self.X.iloc[val, :]
            y_val = self.y.iloc[val]

            self.xgb.fit(X_train, y_train, eval_metric=gini_score)
            self.model_predict(X_val, y_val)
            best_trees.append(self.xgb.best_iteration)

        print("Model Fit")
        self.xgb.fit(self.X, self.y, eval_metric=gini_score)

    def model_gridsearch_fit(self, grid):
        params = self.xgb.get_xgb_params()
        xgb_grid = GridSearchCV(estimator=self.xgb, param_grid=grid, scoring='roc_auc', n_jobs=-1, cv=self.k_fold, iid=False)
        xgb_grid.fit(self.X, self.y)
        print(" >> Grid Scores\n{}".format(xgb_grid.grid_scores_))
        print(" >> Best Params  {}".format(xgb_grid.best_params_))
        print(" >> Best Score   {}".format(xgb_grid.best_score_))

        self.xgb = xgb_grid.estimator
        params = self.xgb.get_params()
        for k, v in xgb_grid.best_params_.items():
            params[k] = v
        self.xgb.set_params(params)
        self.model_fit()

    def model_predict(self):
        y_pred = self.xgb.predict(self.X)
        y_prob = self.xgb.predict_proba(self.X)[:, 1]
        print(gini_normalized(self.y, y_prob))

    def model_test(self):
        y_pred = self.xgb.predict(self.T)
        y_prob = self.xgb.predict_proba(self.T)[:, 1]
        return y_pred, y_prob

    def get_params(self):
        return self.xgb.get_xgb_params()

    def get_model(self):
        return self.xgb


class Stacker:

    def __init__(self, X, y, T, k_fold, stacker_1, stacker_2):
        self.X = X
        self.y = y
        self.T = T
        self.k_fold = k_fold
        self.stacker_1 = stacker_1
        self.stacker_2 = stacker_2

    def model_fit(self):

        folds = list(StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=2020).split(self.X, self.y))
        print(self.X.shape[0], len(self.stacker_1))
        s_train = np.zeros((self.X.shape[0], len(self.stacker_1)))
        s_test = np.zeros((self.T.shape[0], len(self.stacker_1)))

        # Layer 1
        for idx, clf in enumerate(self.stacker_1):

            s_test_temp = np.zeros((self.T.shape[0], self.k_fold))
            for num, (train_idx, val_idx) in enumerate(folds):
                X_train = self.X.loc[train_idx]
                y_train = self.y.loc[train_idx]

                clf.fit(X_train, y_train)
                s_train[val_idx, idx] = clf.predict_proba(self.X.loc[val_idx])[:, 1]
                s_test_temp[:, num] = clf.predict_proba(self.T)[:, 1]
            print(s_test_temp.mean(axis=1))
            s_test[:, idx] = s_test_temp.mean(axis=1)

        print(s_train)
        print(s_test)

        # Layer 2
        self.stacker_2.fit(s_train, self.y)
        y_prob_train = self.stacker_2.predict_proba(s_train)[:, 1]
        y_pred_train = self.stacker_2.predict(s_train)
        y_pred_test  = self.stacker_2.predict_proba(s_test)[:, 1]
        print("[TRAIN Score][AUC]  {}".format(roc_auc_score(self.y, y_pred_train)))
        print("[TRAIN Score][GINI] {}".format(gini_normalized(self.y, y_prob_train)))

        #pd.DataFrame({'target': y_pred_test, 'id': test_df['id']}).to_csv('../dataset/submission.csv', index=False)