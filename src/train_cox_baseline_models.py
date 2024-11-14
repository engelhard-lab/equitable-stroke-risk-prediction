#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

import sys, os

from load_stroke_data import load_stroke_data

DATA_PATH = '../data/stroke_risk_ads_v5i_comprisk.csv'


def main():

    # train baseline models that include race

    print()
    print('Training Cox baselines that include race')
    print()

    data = load_stroke_data(DATA_PATH)

    # code non-stroke events as censoring

    for part in ['s_train', 's_val', 's_test', 's_regards']:
        data[part] = data[part] == 1

    train_cox_models(data, '../results/aim_revision/cox_baselines')

    # train baseline models that do not include race

    print()
    print('Training Cox baselines that do not include race')
    print()

    data = load_stroke_data(DATA_PATH, include_race=False)

    # code non-stroke events as censoring

    for part in ['s_train', 's_val', 's_test', 's_regards']:
        data[part] = data[part] == 1

    train_cox_models(data, '../results/aim_revision/cox_baselines_race_free')


def train_cox_models(data, results_dir):

    create_folder(results_dir)

    rs = np.random.RandomState(7)

    l2_alphas = np.logspace(-4, 0, 13)

    MAX_EPOCHS = 200

    results = []

    for i, lr in enumerate(l2_alphas):
                
        print('Model %i: lambda_l2 = %.3e' % (i, lr))

        model = CoxPHSurvivalAnalysis(alpha=lr)

        model = model.fit(
            data['X_train'],
            Surv.from_arrays(data['s_train'], data['t_train'])
        )

        results.append({
            'lambda_l2': lr,
            'train_ci': model.score(
                data['X_train'],
                Surv.from_arrays(data['s_train'], data['t_train'])
            ),
            'val_ci': model.score(
                data['X_val'],
                Surv.from_arrays(data['s_val'], data['t_val'])
            )
        })

        # SAVE RESULTS FOR VALIDATION, TEST, AND REGARDS SETS

        for X_part, name in zip(
            (data['X_val'], data['X_test'], data['X_regards']),
            ('val', 'test', 'regards')
        ):

            np.save(
                os.path.join(results_dir, 'run_%i_%s_pred.npy' % (i, name)),
                model.predict(X_part)
            )
            
            np.save(
                os.path.join(results_dir, 'run_%i_%s_surv_10yr.npy' % (i, name)),
                np.array([fn(10) for fn in model.predict_survival_function(X_part)])
            )
            
        pd.DataFrame(results).to_csv(
            os.path.join(results_dir, 'training_summary.csv'))


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created.")
    else:
        print(f"Folder '{path}' already exists.")


if __name__ == '__main__':
    main()



