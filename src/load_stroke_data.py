import numpy as np
import pandas as pd


def get_predictors(include_race=True):
    
    ID_COLS = [
        'usubjid', 'original_id', 'study', 
        'exam', 'cohort', 'days_since_exam1'
    ]

    DEM_COLS = [
        'sex', 'age', 'race_c', 
        'educlev', 'fam_income', 'income_cat',
        'state', 'region'
    ]

    VITALS_COLS = [
        'wgt', 'hgt_cm', 'bmi',
        'sysbp', 'diabp'
    ]

    COMORB_COLS = [
        'diab', 'diab_sr', 'hrx', 'hyper_sr',
        'hx_chd', 'hxcvd', 'hxhrtd',
        'hxmi', 'hxmi_sr', 'afib', 'afib_sr', 'lvh',
        'valvdis', 'carsten', 'currsmk',
        'fh_stroke', 'health',
        'hxtia', 'hxpad'
    ]

    LAB_COLS = [
        'tc', 'hdl', 'trig',
        'ldl', 'ldl_calc',
        'fasting_bg', 'creat'
    ]

    MED_COLS = [
        'anycholmed', 'statin', 'nonstatin',
        'statnonstat', 'aspirin', 'insulin'
    ]

    DIET_COLS = [
        'vegetables', 'vegetables_q', 'fruits', 'fruits_q',
        'sodium', 'sodium_q', 'alcohol', 'alcohol_cat',
        'activity', 'activity_alt', 'inactivity'
    ]

    OMIT_COLS = [
        'fam_income',
        'diab_sr',
        'hyper_sr',
        'hx_chd',
        'hxmi_sr',
        'afib_sr',
        'valvdis',
        'carsten',
        'hxtia',
        'hxpad',
        'ldl',
        'nonstatin',
        'statnonstat',
        'vegetables',
        'fruits',
        'sodium',
        'alcohol',
        'inactivity'
    ]

    OMIT_COLS = OMIT_COLS + ['state']
    
    if not include_race:
        OMIT_COLS = OMIT_COLS + ['race_c']
    
    CATEGORICAL_PREDICTORS = [
        'race_c', 'educlev', 'income_cat', 'region', 'health',
        'vegetables_q', 'fruits_q',
        'sodium_q', 'alcohol_cat',
        'activity', 'activity_alt'
    ]

    PREDICTORS = [
        p for p in (
            DEM_COLS + VITALS_COLS +
            COMORB_COLS + LAB_COLS +
            MED_COLS + DIET_COLS
        )
        if p not in OMIT_COLS
    ]
    
    numeric_predictors = [
        p for p in PREDICTORS
        if not p in CATEGORICAL_PREDICTORS
    ]

    categorical_predictors = [
        p for p in PREDICTORS
        if p in CATEGORICAL_PREDICTORS
    ]
    
    return numeric_predictors, categorical_predictors


def load_stroke_data(filename, include_race=True):
    
    # df = pd.read_csv(
    #     '../datasets/stroke_risk_ads_v4i.csv'
    # )
    
    # df = pd.read_csv(
    #     '../datasets/stroke_risk_ads_v5i.csv',
    #     low_memory=False
    # )

    # df = pd.read_csv(
    #     '../data/stroke_risk_ads_v5i_comprisk.csv',
    #     low_memory=False
    # )

    df = pd.read_csv(filename, low_memory=False)
    
    ENDPOINT_COLS = [
        'stroke', 't2stroke', 't2stroke_yrs',
        'stroke10', 't2stroke10', 't2stroke10_yrs',
        'stroke12', 't2stroke12', 't2stroke12_yrs',
        'stroke12_cr1', 't2stroke12_cr1', 't2stroke12_cr1_yrs'
    ]

    numeric_predictors, categorical_predictors = get_predictors(
        include_race=include_race
    )
    
    all_predictors = numeric_predictors + categorical_predictors
    
    if not include_race:
        print('Race will be excluded as a predictor')
        print()
    
    print('The following variables will be one-hot encoded:')
    print(categorical_predictors)
    print()
    
    print('The following variables are numeric:')
    print(numeric_predictors)
    print()
    
    for p in numeric_predictors:
        df.loc[:, p] = pd.to_numeric(df[p], errors='coerce')
    
    if df[numeric_predictors].isna().sum().sum() > 0:
        print('Found invalid values among numeric predictors:')
        print(df[numeric_predictors].isna().sum())
        print()
        invalid_rows = df[numeric_predictors].isna().any(axis=1)
        print('Dropping %i invalid rows.' % np.sum(invalid_rows))
        print()
        df = df[~invalid_rows]

    frame = (
        df[df['study'] != 'REGARDS']
        .sample(frac=1., random_state=2022)
    )

    val_idx = int(len(frame) * .6)
    test_idx = int(len(frame) * .8)

    train_frame = frame[:val_idx]
    val_frame = frame[val_idx:test_idx]
    test_frame = frame[test_idx:]

    regards_frame = (
        df[df['study'] == 'REGARDS']
        .sample(frac=1., random_state=2022)
    )
    
    print(
        'Frame sizes are',
        len(train_frame),
        len(val_frame),
        len(test_frame),
        len(regards_frame)
    )

    # PREPROCESS DATA

    from sklearn.preprocessing import OneHotEncoder

    def preprocess_frames(train_frame, val_frames):

        if len(categorical_predictors) == 0:
            return (f.values for f in ([train_frame] + val_frames))
        
        ohe = (
            OneHotEncoder(sparse=False, handle_unknown='ignore')
            .fit(train_frame[categorical_predictors].values)
        )

        return (
            np.concatenate(
                [
                    ohe.transform(f[categorical_predictors].values),
                    f[numeric_predictors].values
                ],
                axis=1
            )
            for f in ([train_frame] + val_frames)
        )

    X_train, X_val, X_test, X_regards = preprocess_frames(
        train_frame[all_predictors],
        [
            val_frame[all_predictors],
            test_frame[all_predictors],
            regards_frame[all_predictors]
        ]
    )
    
    print('After preprocessing, there are %i predictors' % len(X_train.T))

    Xtr_mean = np.mean(X_train, axis=0)
    Xtr_std = np.std(X_train, axis=0)

    X_train, X_val, X_test, X_regards = (
        ((X - Xtr_mean) / Xtr_std)
        for X in [X_train, X_val, X_test, X_regards]
    )
    
    s_col = 'stroke12_cr1'
    t_col = 't2stroke12_cr1'
    
    s_train, s_val, s_test, s_regards = (
        fr[s_col].values
        for fr in (train_frame, val_frame, test_frame, regards_frame)
    )
    
    t_train, t_val, t_test, t_regards = (
        fr[t_col].values / 366.
        for fr in (train_frame, val_frame, test_frame, regards_frame)
    )
    
    mbv_train, mbv_val, mbv_test, mbv_regards = (
        (fr['race_c'] == 'Black').values
        for fr in (train_frame, val_frame, test_frame, regards_frame)
    )

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'X_regards': X_regards,
        's_train': s_train,
        's_val': s_val,
        's_test': s_test,
        's_regards': s_regards,
        't_train': t_train,
        't_val': t_val,
        't_test': t_test,
        't_regards': t_regards,
        'mbv_train': mbv_train,
        'mbv_val': mbv_val,
        'mbv_test': mbv_test,
        'mbv_regards': mbv_regards,
    }