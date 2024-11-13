#!/usr/bin/env python
# coding: utf-8

import sys, os
import warnings

from load_stroke_data import load_stroke_data

from sksurv.metrics import brier_score, concordance_index_ipcw
from sksurv.util import Surv

from scipy.stats import mannwhitneyu

import pandas as pd
import numpy as np

from tte_measures import xCI, xAUCt, xROCt, ipc_weights

DATA_PATH = '../data/stroke_risk_ads_v5i_comprisk.csv'

def main():

    # Load data (needed to calculate performance)

    data = load_stroke_data(DATA_PATH)

    # code non-stroke events as censoring

    for part in ['s_train', 's_val', 's_test', 's_regards']:
        data[part] = (data[part] == 1).astype(int)

    # Load results summary files

    RESULTS_DIR = '../results/aim_revision'

    calculate_performance_measures(
        data,
        os.path.join(RESULTS_DIR, 'cox_baselines'),
        'Cox'
    )

    calculate_performance_measures(
        data,
        os.path.join(RESULTS_DIR, 'cox_baselines_race_free'),
        'Cox'
    )

    calculate_performance_measures(
        data,
        os.path.join(RESULTS_DIR, 'parity_constrained_race_free'),
        'NN',
        limit=10
    )

    calculate_performance_measures(
        data,
        os.path.join(RESULTS_DIR, 'parity_constrained'),
        'NN',
        limit=10
    )


def calculate_performance_measures(data, results_dir, model_type, limit=None):

    print()
    print('Calculating performance for models in %s' % results_dir)
    print()

    df = pd.read_csv(
        os.path.join(results_dir, 'training_summary.csv')
    ).drop('Unnamed: 0', axis=1).to_dict(orient='records')

    if limit is not None:
        df = df[:limit]

    if model_type == 'Cox':
        run_prefix = 'cox_'
    else:
        run_prefix = None

    results = []

    for i, run_details in enumerate(df):
        
        for part in ['val', 'test', 'regards']:
            
            results_dict = {
                'idx': i,
                'part': part,
                **run_details,
                **eval_by_run_idx(
                    data,
                    i,
                    current_results_path,
                    run_prefix=run_prefix,
                    part=part
                )
            }

            print('Model %i (%s) IPCW CI (all) is %.3f' % (
                i, part, bl_dict['ci_ipcw_10_all']))
            
            results.append(results_dict)
            
    pd.DataFrame(results).to_csv(
        os.path.join(results_dir, 'performance_summary.csv'),
        index=False)


def previous():

    df = pd.read_csv(
        os.path.join(RESULTS_DIR, 'mmd_race_tuning.csv')
    ).drop('Unnamed: 0', axis=1)

    bldf = pd.read_csv(
        os.path.join(RESULTS_DIR, 'cox_baselines/cox_baselines.csv')
    ).drop('Unnamed: 0', axis=1)

    df_nr = pd.read_csv(
        os.path.join(RESULTS_DIR, 'race_free/mmd_race_tuning.csv')
    ).drop('Unnamed: 0', axis=1)

    bldf_nr = pd.read_csv(
        os.path.join(RESULTS_DIR, 'cox_baselines_race_free/cox_baselines.csv')
    ).drop('Unnamed: 0', axis=1)

    # # Evaluate Cox Model with race as a predictor and save results

    baseline_results_df = []
    current_results_path = os.path.join(RESULTS_DIR, 'cox_baselines')

    for i in range(len(bldf)):
        for part in ['val', 'test', 'regards']:
            bl_dict = {
                'idx': i,
                'part': part,
                'lambda_l2': bldf['lambda_l2'][i],
                **eval_by_run_idx(
                    data,
                    i,
                    current_results_path,
                    run_prefix='cox_',
                    part=part
                )
            }
            baseline_results_df.append(bl_dict)
            
    pd.DataFrame(baseline_results_df).to_csv(
        os.path.join(current_results_path, 'cox_performance_summary.csv'),
        index=False)

    # # Evaluate Cox models without race as a predictor and save results

    baseline_results_norace_df = []
    current_results_path = os.path.join(RESULTS_DIR, 'cox_baselines_race_free')

    for i in range(len(bldf_nr)):
        for part in ['val', 'test', 'regards']:
            bl_dict = {
                'idx': i,
                'part': part,
                'lambda_l2': bldf_nr['lambda_l2'][i],
                **eval_by_run_idx(
                    data,
                    i,
                    current_results_path,
                    run_prefix='cox_',
                    part=part
                )
            }
            baseline_results_norace_df.append(bl_dict)
            
    pd.DataFrame(baseline_results_norace_df).to_csv(
        os.path.join(current_results_path, 'cox_performance_summary.csv'),
        index=False)

    # # Evaluate our NN-based models and save results

    model_results_df = []
    current_results_path = RESULTS_DIR

    start_i = 0

    for i in range(start_i, len(df)):
        print('Calculating for model %i' % i, end='\r')
        for part in ['val', 'test', 'regards']:
            bl_dict = {
                'idx': i,
                'part': part,
                'lambda_mmd': df['lambda_mmd'][i],
                'lambda_l2': df['lambda_l2'][i],
                'early_stopping_criterion': df['early_stopping_criterion'][i],
                'learning_rate': df['learning_rate'][i],
                'num_epochs': df['num_epochs'][i],
                'train_loss': df['train_loss'][i],
                'train_nll': df['train_nll'][i],
                'val_loss': df['val_loss'][i],
                'val_nll': df['val_nll'][i],
                **eval_by_run_idx(
                    data, i, RESULTS_DIR, part=part
                )
            }
            print('Model %i (%s) IPCW CI (all) is %.3f' % (i, part, bl_dict['ci_ipcw_10_all']))
            model_results_df.append(bl_dict)
            
    pd.DataFrame(model_results_df).to_csv(
        os.path.join(RESULTS_DIR, 'model_performance_summary.csv'),
        index=False)

    # # Evaluate NN-based models without race as a predictor and save results

    model_results_norace_df = []
    current_results_path = os.path.join(RESULTS_DIR, 'race_free')

    start_i = 0

    for i in range(start_i, len(df_nr)):
        print('Calculating for model %i' % i, end='\r')
        for part in ['val', 'test', 'regards']:
            bl_dict = {
                'idx': i,
                'part': part,
                'lambda_mmd': df_nr['lambda_mmd'][i],
                'lambda_l2': df_nr['lambda_l2'][i],
                'early_stopping_criterion': df_nr['early_stopping_criterion'][i],
                'learning_rate': df_nr['learning_rate'][i],
                'num_epochs': df_nr['num_epochs'][i],
                'train_loss': df_nr['train_loss'][i],
                'train_nll': df_nr['train_nll'][i],
                'val_loss': df_nr['val_loss'][i],
                'val_nll': df_nr['val_nll'][i],
                **eval_by_run_idx(
                    data, i, RESULTS_DIR_NORACE, part=part
                )
            }
            print('Model %i (%s) IPCW CI (all) is %.3f' % (i, part, bl_dict['ci_ipcw_10_all']))
            model_results_norace_df.append(bl_dict)
            
    pd.DataFrame(model_results_norace_df).to_csv(
        os.path.join(current_results_path, 'model_performance_summary.csv'),
        index=False)


# Define functions to evaluate performance

from evaluation import KaplanMeier
from scipy.stats import chi2

def one_calibration(s_test, t_test, surv, times, n_cal_bins=10, return_curves=False):

    N_times = len(times)
    N_pred = len(surv)

    assert N_times == N_pred
    assert N_times > 0

    hs_stats = []
    p_vals = []
    
    op = []
    ep = []

    for s, time in zip(surv, times):

        #try:
            
        predictions = 1 - s

        prediction_order = np.argsort(-predictions)
        predictions = predictions[prediction_order]
        event_times = t_test.copy()[prediction_order]
        event_indicators = (s_test == 1).copy()[prediction_order]

        # Can't do np.mean since split array may be of different sizes.
        binned_event_times = np.array_split(event_times, n_cal_bins)
        binned_event_indicators = np.array_split(event_indicators, n_cal_bins)
        probability_means = [np.mean(x) for x in np.array_split(predictions, n_cal_bins)]

        hosmer_lemeshow = 0

        observed_probabilities = []
        expected_probabilities = []

        for b in range(n_cal_bins):

            prob = probability_means[b]

            km_model = KaplanMeier(binned_event_times[b], binned_event_indicators[b])
            event_probability = 1 - km_model.predict(time)
            bin_count = len(binned_event_times[b])
            
            if prob >= 1.0:
                warnings.warn(
                    "One-Calibration is not well defined: the risk"
                    f"probability of the {b}th bin was {prob}."
                )
                hosmer_lemeshow = np.nan
            
            else:
                hosmer_lemeshow += (bin_count * event_probability - bin_count * prob) ** 2 / (
                    bin_count * prob * (1 - prob)
                )

            observed_probabilities.append(event_probability)
            expected_probabilities.append(prob)

        hs_stats.append(hosmer_lemeshow)
        p_vals.append(1 - chi2.cdf(hosmer_lemeshow, n_cal_bins - 1))

        op.append(observed_probabilities)
        ep.append(expected_probabilities)
            
    if return_curves:
        return np.array(times), np.array(hs_stats), np.array(p_vals), np.array(op), np.array(ep)

    return np.array(times), np.array(hs_stats), np.array(p_vals)


def evaluate_run_performance(s_train, t_train, s_test, t_test, surv_10yr):

    results = {}

    times, bs = brier_score(
        Surv.from_arrays(s_train.astype('bool'), t_train),
        Surv.from_arrays(s_test.astype('bool'), t_test),
        np.array(surv_10yr)[:, np.newaxis],
        [10])

    for t, b in zip(times, bs):
        results['brier_score_%i' % t] = b
        
    times, chisq, pval, observed, expected = one_calibration(
        s_test, t_test,
        [surv_10yr],
        [10],
        n_cal_bins=10,
        return_curves=True
    )

    for t, c, p, op, ep in zip(times, chisq, pval, observed, expected):

        results['onecal_%i_chisq' % t] = c
        results['onecal_%i_pval' % t] = p

        m, c = np.linalg.lstsq(
            np.vstack([ep, np.ones(len(ep))]).T,
            op,
            rcond=None
        )[0]

        results['onecal_%i_slope' % t] = m
        results['onecal_%i_intercept' % t] = c
        
    ci, _, _, _, _ = concordance_index_ipcw(
        Surv.from_arrays(s_train.astype('bool'), t_train),
        Surv.from_arrays(s_test.astype('bool'), t_test),
        -1 * surv_10yr
    )

    results['ci_ipcw_10'] = ci

    return results


def eval_by_run_idx(data, idx, results_dir, run_prefix='', part='val', bootstrap_seed=None):
    
    ## assumes pred year is the final year (which *should* be year 10)

    black = data['mbv_' + part]
    s_part = data['s_' + part]
    t_part = data['t_' + part]

    labels = ['All', 'Black', 'White']

    #pred = np.load(results_dir + run_prefix + 'run_%i_%s_pred.npy' % (idx, part))
    surv_10yr = np.load(
        os.path.join(
            results_dir,
            run_prefix + 'run_%i_%s_surv_10yr.npy' % (idx, part)
        )
    )
    
    if bootstrap_seed is not None:
        indices = np.arange(len(black))
        bootstrap_mask = rs.choice(indices, len(indices), replace=True)
        black = np.array(black)[bootstrap_mask]
        s_part = np.array(s_part)[bootstrap_mask]
        t_part = np.array(t_part)[bootstrap_mask]
        surv_10yr = np.array(surv_10yr)[bootstrap_mask]
    
    if np.amin(surv_10yr) < 0:
        warnings.warn('Warning: %.2f of surv values are <0' % np.mean(surv_10yr < 0))
    
    if np.amax(surv_10yr) > 1:
        warnings.warn('Warning: %.2f of surv values are >1' % np.mean(surv_10yr > 1))
    
    results = {}
    
    try:
        mwu = mannwhitneyu(surv_10yr[black], surv_10yr[~black])
    except Exception as e:
        print(e)
        mwu = [np.nan, np.nan]
    
    results['avg_U_surv'] = mwu[0] / (np.sum(black) * np.sum(~black))
    results['pval_surv'] = mwu[1]
    
    subset_names = ['all', 'black', 'white']

    for subset_idx, subset_mask in enumerate([black | ~black, black, ~black]):
        
        new_results = evaluate_run_performance(
            data['s_train'], data['t_train'],
            s_part[subset_mask], t_part[subset_mask],
            surv_10yr[subset_mask]
        )
        
        results = {**results, **{(k + '_' + subset_names[subset_idx]): v for k, v in new_results.items()}}
    
    # xCI results

    results['CI (ours from xCI)'] = xCI(
        s_part, t_part, -1 * surv_10yr
    )

    for pos_label, pos_group in zip(['_black', '_white'], [black, ~black]):

        for neg_label, neg_group in zip(['_black', '_white'], [black, ~black]):

            label = 'xCI' + pos_label + neg_label

            try:

                results[label] = xCI(
                    s_part, t_part, -1 * surv_10yr,
                    pos_group=pos_group, neg_group=neg_group,
                )

            except Exception as e:

                print(e)
                results[label] = np.nan

    try:
        
        ipcw = ipc_weights(data['s_train'], data['t_train'], s_part, t_part)

        results['CI IPCW (ours from xCI)'] = xCI(
            s_part, t_part, -1 * surv_10yr,
            weights=ipcw
        )

        for pos_label, pos_group in zip(['_black', '_white'], [black, ~black]):

            for neg_label, neg_group in zip(['_black', '_white'], [black, ~black]):

                label = 'xCI_ipcw' + pos_label + neg_label

                results[label] = xCI(
                    s_part, t_part, -1 * surv_10yr,
                    pos_group=pos_group, neg_group=neg_group,
                    weights=ipcw
                )

    except Exception as e:

        print(e)
        results['CI IPCW (ours from xCI)'] = np.nan
        results['xCI_ipcw_black_black'] = np.nan
        results['xCI_ipcw_black_white'] = np.nan
        results['xCI_ipcw_white_black'] = np.nan
        results['xCI_ipcw_white_white'] = np.nan
        
    return results


if __name__ == '__main__':
    main()
