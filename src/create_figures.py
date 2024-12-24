#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from load_stroke_data import load_stroke_data

import sys, os
#sys.path.append('../../tte-performance')
from tte_measures import  xCI, xAUCt, xROCt, xAPt, xPRt, ipc_weights, kaplan_meier


#plt.style.use('seaborn')
sns.set()

DATA_PATH = '../data/stroke_risk_ads_v5i_comprisk.csv'

RESULTS_BASEDIR = '../results/aim_revision'

#RESULTS_DIR = '../results/results_090723/'
#RESULTS_DIR_NORACE = '../results/results_120323_norace/'

# df = pd.read_csv(RESULTS_DIR + 'model_performance_summary.csv').merge(
#     pd.read_csv(RESULTS_DIR + 'model_xCI_summary.csv')
#     #pd.read_csv('model_xCI_df_%s.csv' % RESULTS_DATE)
# )

print('Loading results summaries')

df = pd.read_csv(os.path.join(
    RESULTS_BASEDIR,
    'parity_constrained',
    'performance_summary.csv'
    ))

# bdf = pd.read_csv(RESULTS_DIR + 'cox_performance_summary.csv').merge(
#     pd.read_csv(RESULTS_DIR + 'cox_xCI_summary.csv')
# )

bdf = pd.read_csv(os.path.join(
    RESULTS_BASEDIR,
    'cox_baselines',
    'performance_summary.csv'
    ))

# df_nr = pd.read_csv(RESULTS_DIR_NORACE + 'model_performance_summary.csv').merge(
#     pd.read_csv(RESULTS_DIR_NORACE + 'model_xCI_summary.csv')
#     #pd.read_csv('model_xCI_df_%s.csv' % RESULTS_DATE)
# )

df_nr = pd.read_csv(os.path.join(
    RESULTS_BASEDIR,
    'parity_constrained_race_free',
    'performance_summary.csv'
    ))

# bdf_nr = pd.read_csv(RESULTS_DIR_NORACE + 'cox_performance_summary.csv').merge(
#     pd.read_csv(RESULTS_DIR_NORACE + 'cox_xCI_summary.csv')
# )

bdf_nr = pd.read_csv(os.path.join(
    RESULTS_BASEDIR,
    'cox_baselines_race_free',
    'performance_summary.csv'
    ))


# # xCI results

# In[5]:

print('Finding best performing models')


# use best validation result

for frame in [bdf, df, bdf_nr, df_nr]:
    frame['mean_xCI'] = frame.loc[:, 'xCI_black_black':'xCI_white_white'].mean(axis=1)
    frame['min_xCI'] = frame.loc[:, 'xCI_black_black':'xCI_white_white'].min(axis=1)

    frame['mean_ipcw_xCI'] = frame.loc[:, 'xCI_ipcw_black_black':'xCI_ipcw_white_white'].mean(axis=1)
    frame['min_ipcw_xCI'] = frame.loc[:, 'xCI_ipcw_black_black':'xCI_ipcw_white_white'].min(axis=1)


# In[6]:


CI_COLUMNS = [
    'xCI_black_black', 'xCI_black_white', 'xCI_white_black', 'xCI_white_white',
    'xCI_ipcw_black_black', 'xCI_ipcw_black_white', 'xCI_ipcw_white_black', 'xCI_ipcw_white_white',
    'mean_ipcw_xCI', 'min_ipcw_xCI',
    'CI IPCW (ours from xCI)'
]


# In[7]:


best_cox = bdf[bdf['part'] == 'val'].sort_values('min_ipcw_xCI', ascending=False)['idx'].values[0]
best_cox_nr = bdf_nr[bdf_nr['part'] == 'val'].sort_values('min_ipcw_xCI', ascending=False)['idx'].values[0]


# In[8]:


df['criterion'] = df['min_ipcw_xCI'] + df['CI IPCW (ours from xCI)']

best_no_mmd = df[(df['part'] == 'val') & (df['lambda_mmd'] == 0)].sort_values(
    'CI IPCW (ours from xCI)', ascending=False)['idx'].values[0]
best_mmd = df[(df['part'] == 'val') & (df['lambda_mmd'] > 0)].sort_values(
    'CI IPCW (ours from xCI)', ascending=False)['idx'].values[0]
best_any = df[df['part'] == 'val'].sort_values(
    'CI IPCW (ours from xCI)', ascending=False)['idx'].values[0]

best_nr = df_nr[df_nr['part'] == 'val'].sort_values(
    'CI IPCW (ours from xCI)', ascending=False)['idx'].values[0]

fair_no_mmd = df[(df['part'] == 'val') & (df['lambda_mmd'] == 0)].sort_values(
    'criterion', ascending=False)['idx'].values[0]
fair_mmd = df[(df['part'] == 'val') & (df['lambda_mmd'] > 0)].sort_values(
    'criterion', ascending=False)['idx'].values[0]


# previous result: (0, 249, 1279)

# In[9]:


best_cox, best_any, fair_mmd, best_cox_nr, best_nr


# In[10]:


# df['MMD Status'] = 'No Parity Constraint'
# df.loc[df['lambda_mmd'] > 0, 'MMD Status'] = 'With Parity Constraint'


# # Load dataset (for more detailed results, including bootstrapping)

# In[11]:


# sys.path.append('../src')
# from load_data import load_paper_2_data
# data = load_paper_2_data()

print('Loading data')

data = load_stroke_data(DATA_PATH)

# code non-stroke events as censoring

for part in ['s_train', 's_val', 's_test', 's_regards']:
    data[part] = (data[part] == 1).astype(int)


def load_predictions(idx, results_dir, part='val'):
    
    surv_10yr = np.load(os.path.join(
        results_dir,
        'run_%i_%s_surv_10yr.npy' % (idx, part)
        ))
    
    if np.amin(surv_10yr) < 0:
        warnings.warn('Warning: %.2f of surv values are <0' % np.mean(surv_10yr < 0))
    
    if np.amax(surv_10yr) > 1:
        warnings.warn('Warning: %.2f of surv values are >1' % np.mean(surv_10yr > 1))
        
    return 1 - surv_10yr

cox_pred_test = load_predictions(
    best_cox,
    os.path.join(RESULTS_BASEDIR, 'cox_baselines'),
    part='test'
)

cox_pred_regards = load_predictions(
    best_cox,
    os.path.join(RESULTS_BASEDIR, 'cox_baselines'),
    part='regards'
)

best_no_mmd_pred_test = load_predictions(
    best_no_mmd,
    os.path.join(RESULTS_BASEDIR, 'parity_constrained'),
    part='test'
)

best_no_mmd_pred_regards = load_predictions(
    best_no_mmd,
    os.path.join(RESULTS_BASEDIR, 'parity_constrained'),
    part='regards'
)

best_mmd_pred_test = load_predictions(
    best_mmd,
    os.path.join(RESULTS_BASEDIR, 'parity_constrained'),
    part='test'
)

best_mmd_pred_regards = load_predictions(
    best_mmd,
    os.path.join(RESULTS_BASEDIR, 'parity_constrained'),
    part='regards'
)

fair_no_mmd_pred_test = load_predictions(
    fair_no_mmd,
    os.path.join(RESULTS_BASEDIR, 'parity_constrained'),
    part='test'
)

fair_no_mmd_pred_regards = load_predictions(
    fair_no_mmd,
    os.path.join(RESULTS_BASEDIR, 'parity_constrained'),
    part='regards'
)

fair_mmd_pred_test = load_predictions(
    fair_mmd,
    os.path.join(RESULTS_BASEDIR, 'parity_constrained'),
    part='test'
)

fair_mmd_pred_regards = load_predictions(
    fair_mmd,
    os.path.join(RESULTS_BASEDIR, 'parity_constrained'),
    part='regards'
)

cox_nr_pred_test = load_predictions(
    best_cox_nr,
    os.path.join(RESULTS_BASEDIR, 'cox_baselines_race_free'),
    part='test'
)

cox_nr_pred_regards = load_predictions(
    best_cox_nr,
    os.path.join(RESULTS_BASEDIR, 'cox_baselines_race_free'),
    part='regards'
)

best_nr_pred_test = load_predictions(
    best_nr,
    os.path.join(RESULTS_BASEDIR, 'parity_constrained_race_free'),
    part='test'
)

best_nr_pred_regards = load_predictions(
    best_nr,
    os.path.join(RESULTS_BASEDIR, 'parity_constrained_race_free'),
    part='regards'
)

COX_DIR = os.path.join(RESULTS_BASEDIR, 'cox_baselines')
COXRF_DIR = os.path.join(RESULTS_BASEDIR, 'cox_baselines_race_free')
PC_DIR = os.path.join(RESULTS_BASEDIR, 'parity_constrained')
PCRF_DIR = os.path.join(RESULTS_BASEDIR, 'parity_constrained_race_free')


# # Define variables to be used across several different figures

set_labels = ['Performance on the Test Set', 'Performance on REGARDS']

fig1_summary = []

# model_labels = [
#     'Cox Model', 'Best Model',
#     'Race-Free Cox Model', 'Race-Free Best Model',
#     'Fair Model (by xCI)'
# ]

model_labels = [
    'Cox Model', 'Unconstrained Model',
    'Race-Free Cox Model', 'Race-Free Unconstrained Model',
    'Parity-Constrained Model'
]

model_idx = [best_cox, best_any, best_cox_nr, best_nr, fair_mmd]

model_dir = [COX_DIR, PC_DIR, COXRF_DIR, PCRF_DIR, PC_DIR]

legend_colors = ['b', 'g', 'c', 'm', 'y']

x_labels = [
    'Black Cases,\nBlack Controls',
    'White Cases,\nWhite Controls',
    'Black Cases,\nWhite Controls',
    'White Cases,\nBlack Controls',
]


# # Figure 1

print('Creating Performance Summary Figure')

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

for a, part, slbl in zip(ax, ['test', 'regards'], set_labels):
    
    for offset, idx, lbl, mdir, c in zip(
        [-.3, -.15, 0, .15, .3], model_idx, model_labels, model_dir, legend_colors):
        
        # plot the overall CI
        
        pred_risk = load_predictions(idx, mdir, part=part)
        
        s_part = data['s_' + part]
        t_part = data['t_' + part]
        black = data['mbv_' + part]
        
        weights = ipc_weights(
            data['s_train'], data['t_train'],
            s_part, t_part
        )
        
        estimate = xCI(
            s_part, t_part, pred_risk, weights=weights
        )
        
        a.plot([-.5, 3.5], [estimate, estimate], '--.', color=c)
        
        estimates = []
        cis = []
        
        for g1, g2 in zip(
            [black, ~black, black, ~black],
            [black, ~black, ~black, black]
        ):
            
            estimate, ci_low, ci_high = xCI(
                s_part, t_part, pred_risk, weights=weights,
                pos_group=g1, neg_group=g2, n_bootstrap_samples=100
            )
            
            estimates.append(estimate)
            cis.append([estimate - ci_low, ci_high - estimate])
        
        a.bar(
            np.arange(4) + offset, estimates,
            yerr=np.array(cis).T,
            width=.15, label=lbl, color=c
        )
        
        fig1_summary.append({
            'set': part,
            'mdl_idx': idx,
            'mdl_lbl': lbl,
            'mdl_dir': mdir,
            'mean_bb': estimates[0],
            'ci_bb_low': estimates[0] - cis[0][0],
            'ci_bb_high': cis[0][1] + estimates[0],
            'mean_ww': estimates[1],
            'ci_ww_low': estimates[1] - cis[1][0],
            'ci_ww_high': cis[1][1] + estimates[1],
            'mean_bw': estimates[2],
            'ci_bw_low': estimates[2] - cis[2][0],
            'ci_bw_high': cis[2][1] + estimates[2],
            'mean_wb': estimates[3],
            'ci_wb_low': estimates[3] - cis[3][0],
            'ci_wb_high': cis[3][1] + estimates[3]
        })
        
    a.set_xticks(np.arange(4))
    a.set_xticklabels(x_labels)
    a.plot([0, 1], [1.1, 1.1], 'k--', label='IPCW CI (All Cases and Controls)')
    a.legend()
    a.set_ylim([0.5, 1.])
    a.set_xlim([-.5, 3.5])
    a.set_ylabel('IPCW xCI')
    a.set_title(slbl)
    
pd.DataFrame(fig1_summary).to_csv('../figures/fig1_summary.csv', index=False)

plt.tight_layout()
plt.savefig('../figures/fig1.pdf')
plt.show()


# # Figure 2: ROC at 10 years

print('Creating ROC Curves')

# load models (Cox, best, fairest) x (test set, REGARDS)

time = 10
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30, 12), sharex=True, sharey=True)

for rowidx, part in enumerate(['test', 'regards']):

    for colidx, (idx, lbl, mdir) in enumerate(zip(model_idx, model_labels, model_dir)):

        pred_risk = load_predictions(idx, mdir, part=part)

        s_part = data['s_' + part]
        t_part = data['t_' + part]
        black = data['mbv_' + part]
        
        a = ax[rowidx, colidx]

        for _, (g1, l1, g2, l2) in enumerate(zip(
            [black, ~black, black, ~black],
            ['Black', 'White', 'Black', 'White'],
            [black, ~black, ~black, black],
            ['Black', 'White', 'White', 'Black']
        )):
            
            print('Running for %s (%s-%s) on %s' % (lbl, l1, l2, part.capitalize()))

            auct, ci_low, ci_high = xAUCt(
                s_part, t_part, pred_risk, np.array([time]),
                pos_group=g1,
                neg_group=g2,
                n_bootstrap_samples=50
            )

            tprt, fprt, _ = xROCt(
                s_part, t_part, pred_risk, time,
                pos_group=g1,
                neg_group=g2
            )

            a.plot(
                fprt, tprt,
                label='%s-%s\nxAUC$_t$ = %.2f (%.2f-%.2f)' % (
                    l1, l2, auct[0], ci_low[0], ci_high[0]
                )
            )
            
        a.set_title(lbl + ' on %s' % part.capitalize())

        a.plot([0, 1], [0, 1], 'k--', label='No information')
        a.set_ylabel('True Positive Rate (t=%i)' % time)
        a.set_xlabel('False Positive Rate (t=%i)' % time)

        a.legend()

plt.tight_layout()
plt.savefig('../figures/roc.pdf')
plt.show()


# # Fig 2b: PR Curve at 10 years

print('Creating PR Curves')

# load models (Cox, best, fairest) x (test set, REGARDS)
time = 10
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30, 12), sharex=True, sharey=True)

for rowidx, part in enumerate(['test', 'regards']):

    for colidx, (idx, lbl, mdir) in enumerate(zip(model_idx, model_labels, model_dir)):

        pred_risk = load_predictions(idx, mdir, part=part)

        s_part = data['s_' + part]
        t_part = data['t_' + part]
        black = data['mbv_' + part]

        for _, (g1, l1, g2, l2) in enumerate(zip(
            [black, ~black, black, ~black],
            ['Black', 'White', 'Black', 'White'],
            [black, ~black, ~black, black],
            ['Black', 'White', 'White', 'Black']
        )):
            
            print('Running for %s (%s-%s) on %s' % (lbl, l1, l2, part.capitalize()))

            a = ax[rowidx, colidx]

            (apt, prevt), (apt_low, prev_low), (apt_high, prev_high) = xAPt(
                s_part, t_part, pred_risk, np.array([time]),
                pos_group=g1,
                neg_group=g2,
                n_bootstrap_samples=50,
                return_prevalence=True
            )

            recallt, precisiont, _, prevt = xPRt(
                s_part, t_part, pred_risk, time,
                pos_group=g1,
                neg_group=g2
            )

            p = a.plot(
                recallt, precisiont,
                label='%s-%s\nxAP$_t$ = %.2f (%.2f-%.2f)' % (
                    l1, l2, apt[0], apt_low[0], apt_high[0]
                )
            )
            
            a.plot([0, 1], [prevt, prevt], '--', color=p[0].get_color())
            
            a.set_title(lbl + ' on %s' % part.capitalize())
            
            a.set_ylabel('Precision (t=%i)' % time)
            a.set_xlabel('Recall (t=%i)' % time)
            
            a.legend()

plt.tight_layout()
plt.savefig('../figures/pr.pdf')
plt.show()


# # Figure: CI (all) vs Min xCI (v2)

print('Creating model density')

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

frames = [df[df['lambda_mmd'] > 0], df[df['lambda_mmd'] == 0]]
labels = ['No Parity Constraint', 'With Parity Constraint']
set_labels = ['Validation Set', 'Test Set', 'REGARDS']
colors = ['blue', 'red']

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

for a, part, slbl in zip(axs, ['val', 'test', 'regards'], set_labels):
        
    #sns.histplot(
    sns.kdeplot(
        x=df[(df['lambda_mmd'] > 0) & (df['part'] == part)]['CI IPCW (ours from xCI)'],
        y=df[(df['lambda_mmd'] > 0) & (df['part'] == part)]['min_ipcw_xCI'],
        ax=a,
        label='With Fairness Constraint',
        #binwidth=[.01, .01],
        bw_adjust=.35,
        gridsize=100,
        levels=5,
        color=colors[0],
        alpha=.5,
    )
    
    #sns.histplot(
    sns.kdeplot(
        x=df[(df['lambda_mmd'] == 0) & (df['part'] == part)]['CI IPCW (ours from xCI)'],
        y=df[(df['lambda_mmd'] == 0) & (df['part'] == part)]['min_ipcw_xCI'],
        ax=a,
        label='No Fairness Constraint',
        #binwidth=[.01, .01],
        bw_adjust=.35,
        gridsize=100,
        levels=5,
        color=colors[1],
        alpha=.5,
    )
        
    cox_ci = bdf[(bdf['part'] == part) & (bdf['idx'] == best_cox)]['CI IPCW (ours from xCI)']
    cox_min_xci = bdf[(bdf['part'] == part) & (bdf['idx'] == best_cox)]['min_ipcw_xCI']

    best_ci = df[(df['part'] == part) & (df['idx'] == best_any)]['CI IPCW (ours from xCI)']
    best_min_xci = df[(df['part'] == part) & (df['idx'] == best_any)]['min_ipcw_xCI']

    balanced_ci = df[(df['part'] == part) & (df['idx'] == fair_mmd)]['CI IPCW (ours from xCI)']
    balanced_min_xci = df[(df['part'] == part) & (df['idx'] == fair_mmd)]['min_ipcw_xCI']

    a.plot(
        [.5, .85],
        [cox_min_xci, cox_min_xci],
        'k--',
        label='Cox Model'
    )

    a.plot(
        [cox_ci, cox_ci],
        [.5, .85],
        'k--'
    )
    
    a.plot(
        [0., 1.],
        [0., 1.],
        'k-.',
        label='Fairness Ideal (CI = min xCI)'
    )

    if True:#lbl == labels[-1]:
        
        a.scatter(
            balanced_ci,
            balanced_min_xci,
            label='Fairest Model',
            s=100,
            c=colors[0]
        )

        a.scatter(
            best_ci,
            best_min_xci,
            label='Strongest Model',
            s=100,
            c=colors[1]
        )

    a.set_xlim([0.54, 0.80])
    a.set_ylim([0.50, 0.76])
    a.set_xlabel('Concordance Index', fontsize=16)
    a.set_ylabel('Minimum xCI', fontsize=16)
    a.set_title(slbl, fontsize=16)
    
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor=colors[0], label='With Parity Constraint', alpha=.5),
        Patch(facecolor=colors[1], edgecolor=colors[1], label='Without Parity Constraint', alpha=.5),
        Line2D([0], [0], color='k', lw=1, linestyle='-.', label='Equal Inter-Group xCIs (CI = min xCI)'),
        Line2D([0], [0], color='k', lw=1, linestyle='--', label='Cox Model'),
        Line2D([0], [0], marker='o', color='w', label='Parity-Constrained Model', markerfacecolor=colors[0], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Unconstrained Model', markerfacecolor=colors[1], markersize=10),
    ]

    a.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig('../figures/fig3.pdf')
plt.show()


# # Predictive distributions

print('Creating predictive distributions')

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), sharey=True)

for rowidx, part in enumerate(['test', 'regards']):

    for colidx, (idx, lbl, mdir) in enumerate(zip(model_idx, model_labels, model_dir)):

        pred_risk = load_predictions(idx, mdir, part=part)

        s_part = data['s_' + part]
        t_part = data['t_' + part]
        black = data['mbv_' + part].astype(bool)
            
        print('Running for %s on %s' % (lbl, part.capitalize()))

        a = ax[rowidx, colidx]

        pred_year = 10
        stride = 1
        cal_bw = .1

        sns.boxplot(x=~black, y=pred_risk, showfliers=False, ax=a, width=.7)
        a.set_xticklabels(['Black', 'White'])
        
        risk_black = 1 - kaplan_meier(s_part[black], t_part[black], np.array([pred_year, ]))
        risk_white = 1 - kaplan_meier(s_part[~black], t_part[~black], np.array([pred_year, ]))
        
        start, stop = a.get_xlim()
        
        a.plot([start, (stop + start) / 2], [risk_black, risk_black], 'k--')
        a.plot([(stop + start) / 2, stop], [risk_white, risk_white], 'k--', label='10-year prevalence')

        a.set_title(lbl + ' on %s' % part.capitalize())
        a.set_ylabel('Predicted 10-year risk')

        a.legend()

plt.tight_layout()
plt.savefig('../figures/predictive_distributions.pdf')
plt.show()


# # Label-specific predictive distributions

print('Creating label-specific predictive distributions')

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), sharey=True)

for rowidx, part in enumerate(['test', 'regards']):

    for colidx, (idx, lbl, mdir) in enumerate(zip(model_idx, model_labels, model_dir)):

        pred_risk = load_predictions(idx, mdir, part=part)

        s_part = data['s_' + part]
        t_part = data['t_' + part]
        black = data['mbv_' + part].astype(bool)
            
        print('Running for %s on %s' % (lbl, part.capitalize()))

        a = ax[rowidx, colidx]

        pred_year = 10
        stride = 1
        cal_bw = .1

        sns.boxplot(x=s_part, y=pred_risk, hue=~black, showfliers=False, ax=a, width=.7)
        a.set_xticklabels(['Censored', 'Observed\nEvent'])
        
        risk_black = 1 - kaplan_meier(s_part[black], t_part[black], np.array([pred_year, ]))
        risk_white = 1 - kaplan_meier(s_part[~black], t_part[~black], np.array([pred_year, ]))
        
        start, stop = a.get_xlim()
        
        a.set_title(lbl + ' on %s' % part.capitalize())
        a.set_ylabel('Predicted 10-year risk')

        lgnd = a.legend()
        
        lgnd.get_texts()[0].set_text('Black')
        lgnd.get_texts()[1].set_text('White')

plt.tight_layout()
plt.savefig('../figures/predictive_distributions_by_outcome.pdf')
plt.show()

# # calibration

print('Creating calibration plots')

## TODO: FIX THIS

# for calibration, don't code non-stroke events as censoring
# we will use the Aalen-Johansen estimator

data = load_stroke_data(DATA_PATH)

# import os
# sys.path.append(os.path.expanduser('~/dnmc/src'))
# from evaluation import one_calibration, s_cal

from calculate_performance_measures import one_calibration

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))

for rowidx, part in enumerate(['test', 'regards']):

    for colidx, (idx, lbl, mdir) in enumerate(zip(model_idx, model_labels, model_dir)):

        pred_risk = load_predictions(idx, mdir, part=part)

        s_part = data['s_' + part]
        t_part = data['t_' + part]
        black = data['mbv_' + part].astype(bool)

        for _, (subset_mask, l1) in enumerate(zip(
            [black | ~black, black, ~black],
            ['All', 'Black', 'White']
        )):
            
            print('Running for %s (%s) on %s' % (lbl, l1, part.capitalize()))

            a = ax[rowidx, colidx]
            
            pred_year = 10
            stride = 1
            cal_bw = .1
            
            pp, op = s_cal(
                s_part[subset_mask], t_part[subset_mask],
                pred_risk[subset_mask],
                pred_year,
                calibration_bw=cal_bw,
                stride=stride
            )
            
            p = a.plot(pp, op, alpha=.7)
            color = p[0].get_color()

            sns.regplot(x=pp, y=op, ax=a, color=color, scatter=False, label=l1)#, scatter_kws={'s':2})#, marker='.')

            a.plot(
                [0, max(list(pp) + list(op))],
                [0, max(list(pp) + list(op))],
                'k--',
            )

        a.set_xlabel('Predicted')
        a.set_ylabel('Observed')

        a.set_title(lbl + ' on %s' % part.capitalize())

        a.legend()

plt.tight_layout()
plt.savefig('../figures/calibration.pdf')
plt.show()
