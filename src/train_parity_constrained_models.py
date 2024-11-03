#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(2021)

import sys, os

from model import Fair_DTFT, train_model
from evaluation import evaluate_all

# SET RESULTS FOLDER

RESULTS_DIR = '../results/aim_revision/'
DATA_PATH = '../data/stroke_risk_ads_v5i_comprisk.csv'

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created.")
    else:
        print(f"Folder '{path}' already exists.")

create_folder(RESULTS_DIR)

# LOAD DATA

from load_stroke_data import load_stroke_data
data = load_stroke_data(DATA_PATH)

# # NSurv with MMD by race at variable strength

rs = np.random.RandomState(7)

mmd_alphas = [
    0, 0, 0, 0, 1, 1, 1, 10, 10, 10, 100, 100, 1000, 10000
]# + list(np.logspace(0, 4, 9))
l2_alphas = np.logspace(-5, 0, 100)
es_vals = [1, 2, 3]
learning_rates = np.logspace(-4, -1, 100)

MAX_EPOCHS = 200

results = []
i = 0

NUM_RUNS = 2000

for i in range(NUM_RUNS):
    
    ld = np.random.choice(mmd_alphas)
    lr = np.random.choice(l2_alphas)
    es = np.random.choice(es_vals)
    learning_rate = np.random.choice(learning_rates)
            
    print('Model %i: lambda_mmd=%.3e, lambda_l2=%.3e, es_criterion=%i, learning_rate=%.3e' % (i, ld, lr, es, learning_rate))

    model = Fair_DTFT(bins=np.arange(13), n_event_types=2, lr=lr, ld=ld)

    num_epochs, train_loss, train_nll, val_loss, val_nll = train_model(
        model,
        (data['X_train'], data['t_train'], data['s_train'], data['mbv_train']),
        (data['X_val'], data['t_val'], data['s_val'], data['mbv_val']),
        MAX_EPOCHS, batch_size=1000,
        learning_rate=learning_rate,#5e-4,
        early_stopping_criterion=es
    )
    
    results.append({
        'lambda_mmd': ld,
        'lambda_l2': lr,
        'early_stopping_criterion': es,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'train_loss': train_loss,
        'train_nll': train_nll,
        'val_loss': val_loss,
        'val_nll': val_nll
    })

    # SAVE RESULTS FOR VALIDATION, TEST, AND REGARDS SETS

    for X_part, name in zip((data['X_val'], data['X_test'], data['X_regards']), ('val', 'test', 'regards')):

        np.save(
            RESULTS_DIR + 'run_%i_%s_pred.npy' % (i, name),
            model.predict(X_part).numpy()
        )

        np.save(
            RESULTS_DIR + 'run_%i_%s_surv_10yr.npy' % (i, name),
            model.predict_survival_function(X_part, 10).numpy()
        )
        
    pd.DataFrame(results).to_csv(RESULTS_DIR + 'mmd_race_tuning.csv')
