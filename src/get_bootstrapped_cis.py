#!/usr/bin/env python
# coding: utf-8

import sys, os
import warnings

from load_stroke_data import load_stroke_data

from calculate_performance_measures import eval_by_run_idx

from scipy.stats import mannwhitneyu

import pandas as pd
import numpy as np

DATA_PATH = '../data/stroke_risk_ads_v5i_comprisk.csv'

# Load data (needed to calculate performance)

data = load_stroke_data(DATA_PATH)

# Load results summary files

RESULTS_DIR = '../results/aim_revision/'

df = pd.read_csv(
    RESULTS_DIR + 'mmd_race_tuning.csv'
).drop('Unnamed: 0', axis=1)

bldf = pd.read_csv(
    RESULTS_DIR + 'cox_baselines/cox_baselines.csv'
).drop('Unnamed: 0', axis=1)

df_nr = pd.read_csv(
    RESULTS_DIR + 'race_free/mmd_race_tuning.csv'
).drop('Unnamed: 0', axis=1)

bldf_nr = pd.read_csv(
    RESULTS_DIR + 'cox_baselines_race_free/cox_baselines.csv'
).drop('Unnamed: 0', axis=1)

# Identify best models based on validation set results

for frame in [bdf, df, bdf_nr, df_nr]:
    frame['mean_xCI'] = frame.loc[:, 'xCI_black_black':'xCI_white_white'].mean(axis=1)
    frame['min_xCI'] = frame.loc[:, 'xCI_black_black':'xCI_white_white'].min(axis=1)

    frame['mean_ipcw_xCI'] = frame.loc[:, 'xCI_ipcw_black_black':'xCI_ipcw_white_white'].mean(axis=1)
    frame['min_ipcw_xCI'] = frame.loc[:, 'xCI_ipcw_black_black':'xCI_ipcw_white_white'].min(axis=1)

best_cox = bdf[bdf['part'] == 'val'].sort_values('min_ipcw_xCI', ascending=False)['idx'].values[0]

print('The best Cox model is model %i' % best_cox)

best_cox_nr = bdf_nr[bdf_nr['part'] == 'val'].sort_values('min_ipcw_xCI', ascending=False)['idx'].values[0]

print('The best race-free Cox model is model %i' % best_cox_nr)

df['criterion'] = df['min_ipcw_xCI'] + df['CI IPCW (ours from xCI)']

best_no_mmd = df[(df['part'] == 'val') & (df['lambda_mmd'] == 0)].sort_values(
    'CI IPCW (ours from xCI)', ascending=False)['idx'].values[0]
best_mmd = df[(df['part'] == 'val') & (df['lambda_mmd'] > 0)].sort_values(
    'CI IPCW (ours from xCI)', ascending=False)['idx'].values[0]
best_any = df[df['part'] == 'val'].sort_values(
    'CI IPCW (ours from xCI)', ascending=False)['idx'].values[0]

print('The best model is model %i' % best_any)

best_nr = df_nr[df_nr['part'] == 'val'].sort_values(
    'CI IPCW (ours from xCI)', ascending=False)['idx'].values[0]

print('The best race-free model is model %i' % best_nr)

fair_no_mmd = df[(df['part'] == 'val') & (df['lambda_mmd'] == 0)].sort_values(
    'criterion', ascending=False)['idx'].values[0]

print('The fairest model without parity constraint is %i' % fair_no_mmd)

fair_mmd = df[(df['part'] == 'val') & (df['lambda_mmd'] > 0)].sort_values(
    'criterion', ascending=False)['idx'].values[0]

print('The fairest model with parity constraint is %i' % fair_mmd)


# # Get bootstrapped confidence intervals for final models only

# ADDED FOR FINAL RESULTS: BOOTSTRAPPING RESULTS ONLY FOR FINAL MODELS

cox_model_bootstrapped = []

model_idx = best_cox
rs = np.random.RandomState(1999)

for part in ['val', 'test', 'regards']:
    for bootstrap_sample in range(100):
        bl_dict = {
            'idx': model_idx,
            'part': part,
            'lambda_l2': bldf['lambda_l2'][model_idx],
            'bootstrap_sample': bootstrap_sample,
            **eval_by_run_idx(
                model_idx, RESULTS_DIR, run_prefix='cox_', part=part, bootstrap_seed=rs
            )
        }
        cox_model_bootstrapped.append(bl_dict)
        
pd.DataFrame(cox_model_bootstrapped).to_csv(
    RESULTS_DIR + 'cox_baselines/best_cox_bootstrapped.csv', index=False)


# ADDED FOR FINAL RESULTS: BOOTSTRAPPING RESULTS ONLY FOR FINAL MODELS

best_model_bootstrapped = []

model_idx = best_any
rs = np.random.RandomState(2000)

start_i = 0

for part in ['val', 'test', 'regards']:
    for bootstrap_sample in range(100):
        bl_dict = {
            'idx': model_idx,
            'part': part,
            'lambda_mmd': df['lambda_mmd'][model_idx],
            'lambda_l2': df['lambda_l2'][model_idx],
            'early_stopping_criterion': df['early_stopping_criterion'][model_idx],
            'learning_rate': df['learning_rate'][model_idx],
            'num_epochs': df['num_epochs'][model_idx],
            'train_loss': df['train_loss'][model_idx],
            'train_nll': df['train_nll'][model_idx],
            'val_loss': df['val_loss'][model_idx],
            'val_nll': df['val_nll'][model_idx],
            'bootstrap_sample': bootstrap_sample,
            **eval_by_run_idx(
                model_idx, RESULTS_DIR, part=part, bootstrap_seed=rs
            )
        }
        best_model_bootstrapped.append(bl_dict)
        
pd.DataFrame(best_model_bootstrapped).to_csv(
    RESULTS_DIR + 'best_model_bootstrapped.csv', index=False)


# ADDED FOR FINAL RESULTS: BOOTSTRAPPING RESULTS ONLY FOR FINAL MODELS

fair_model_bootstrapped = []

model_idx = fair_mmd
rs = np.random.RandomState(2000)

start_i = 0

for part in ['val', 'test', 'regards']:
    for bootstrap_sample in range(100):
        bl_dict = {
            'idx': model_idx,
            'part': part,
            'lambda_mmd': df['lambda_mmd'][model_idx],
            'lambda_l2': df['lambda_l2'][model_idx],
            'early_stopping_criterion': df['early_stopping_criterion'][model_idx],
            'learning_rate': df['learning_rate'][model_idx],
            'num_epochs': df['num_epochs'][model_idx],
            'train_loss': df['train_loss'][model_idx],
            'train_nll': df['train_nll'][model_idx],
            'val_loss': df['val_loss'][model_idx],
            'val_nll': df['val_nll'][model_idx],
            'bootstrap_sample': bootstrap_sample,
            **eval_by_run_idx(
                model_idx, RESULTS_DIR, part=part, bootstrap_seed=rs
            )
        }
        fair_model_bootstrapped.append(bl_dict)
        
pd.DataFrame(fair_model_bootstrapped).to_csv(
    RESULTS_DIR + 'fair_model_bootstrapped.csv', index=False)


# BOOTSTRAPPING FOR RACE-FREE MODELS

cox_model_norace_bootstrapped = []

model_idx = best_cox_nr

rs = np.random.RandomState(1999)

for part in ['val', 'test', 'regards']:
    for bootstrap_sample in range(100):
        bl_dict = {
            'idx': model_idx,
            'part': part,
            'lambda_l2': bldf_nr['lambda_l2'][model_idx],
            'bootstrap_sample': bootstrap_sample,
            **eval_by_run_idx(
                model_idx, RESULTS_DIR_NORACE, run_prefix='cox_', part=part, bootstrap_seed=rs
            )
        }
        cox_model_norace_bootstrapped.append(bl_dict)
        
pd.DataFrame(cox_model_norace_bootstrapped).to_csv(
    RESULTS_DIR + 'cox_baselines_race_free/best_cox_bootstrapped.csv', index=False)


model_idx = best_nr

best_model_norace_bootstrapped = []

rs = np.random.RandomState(2000)

start_i = 0

for part in ['val', 'test', 'regards']:
    for bootstrap_sample in range(100):
        bl_dict = {
            'idx': model_idx,
            'part': part,
            'lambda_mmd': df_nr['lambda_mmd'][model_idx],
            'lambda_l2': df_nr['lambda_l2'][model_idx],
            'early_stopping_criterion': df_nr['early_stopping_criterion'][model_idx],
            'learning_rate': df_nr['learning_rate'][model_idx],
            'num_epochs': df_nr['num_epochs'][model_idx],
            'train_loss': df_nr['train_loss'][model_idx],
            'train_nll': df_nr['train_nll'][model_idx],
            'val_loss': df_nr['val_loss'][model_idx],
            'val_nll': df_nr['val_nll'][model_idx],
            'bootstrap_sample': bootstrap_sample,
            **eval_by_run_idx(
                model_idx, RESULTS_DIR_NORACE, part=part, bootstrap_seed=rs
            )
        }
        best_model_norace_bootstrapped.append(bl_dict)
        
pd.DataFrame(best_model_norace_bootstrapped).to_csv(
    RESULTS_DIR + 'race_free/best_model_bootstrapped.csv', index=False)
