import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('../eval_scripts')
from models import LightGBMModelWrapper
from core import pattern_to_features

def load_mimic_data():
    df = pd.read_csv('../processed_datasets/mimic_processed.csv')
    y = df['ARDS_FLAG']
    anchor_age = df['anchor_age'].values
    
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'anchor_age', 'ARDS_FLAG']]
    series_names = []
    for col in feature_cols:
        if '_hour_' in col:
            series_name = col.split('_hour_')[0]
            if series_name not in series_names:
                series_names.append(series_name)
    
    X_list = []
    for series_name in series_names:
        series_cols = [col for col in feature_cols if col.startswith(f"{series_name}_hour_")]
        series_cols.sort(key=lambda x: int(x.split('_hour_')[1]))
        X_series = df[series_cols]
        X_list.append(X_series)
    
    return X_list, y, anchor_age, series_names

def load_azt1d_data(subject_id='10'):
    df = pd.read_parquet(f'../processed_datasets/azt1d/subject_{subject_id}.parquet', engine='fastparquet')
    y = df['target']
    cgm_data = df[[col for col in df.columns if col.startswith('CGM_')]]
    insulin_data = df[[col for col in df.columns if col.startswith('Insulin_')]]
    carbs_data = df[[col for col in df.columns if col.startswith('Carbs_')]]
    X_list = [cgm_data, insulin_data, carbs_data]
    series_names = ['CGM', 'Insulin', 'Carbs']
    cgm0 = cgm_data['CGM_0'].values
    return X_list, y, cgm0, series_names

def process_mimic():
    X_list, y, anchor_age, series_names = load_mimic_data()
    
    # Try to load actual pattern parameters, fall back to representative example
    with open('../json_files/mimic/pattern_parameters.json', 'r') as f:
        all_patterns = json.load(f)
    fold_1_patterns = all_patterns['fold_1']

    
    kfold = StratifiedKFold(5, shuffle=True, random_state=42)
    train_idx, val_idx = list(kfold.split(pd.concat(X_list, axis=1), y))[0]
    
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    input_series_train = [x.iloc[train_idx] for x in X_list]
    input_series_test = [x.iloc[val_idx] for x in X_list]
    
    for i in range(len(input_series_train)):
        scaler = StandardScaler()
        input_series_train[i] = pd.DataFrame(scaler.fit_transform(input_series_train[i].values), 
                                             columns=input_series_train[i].columns, index=input_series_train[i].index)
        input_series_test[i] = pd.DataFrame(scaler.transform(input_series_test[i].values), 
                                            columns=input_series_test[i].columns, index=input_series_test[i].index)
    
    age_scaler = StandardScaler()
    train_init = age_scaler.fit_transform(anchor_age[train_idx].reshape(-1, 1))
    test_init = age_scaler.transform(anchor_age[val_idx].reshape(-1, 1))
    
    input_series_train_stacked = np.stack([x.values for x in input_series_train], axis=1)
    input_series_test_stacked = np.stack([x.values for x in input_series_test], axis=1)
    
    train_features, test_features = [train_init], [test_init]
    
    for pattern in fold_1_patterns:
        data_min, data_max = np.min(input_series_train_stacked), np.max(input_series_train_stacked)
        width = int(pattern.get('width', 10))
        use_relative = pattern.get('use_relative', False)
        shift_tolerance = pattern.get('shift_tolerance', 0.0)
        start = pattern.get('start', int(pattern.get('center', 0) - width/2))
        
        # Create pattern array from pattern values or control points
        if 'pattern' in pattern and len(pattern['pattern']) > 0:
            pattern_array = np.array(pattern['pattern'])
            if len(pattern_array) != width:
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(pattern_array))
                x_new = np.linspace(0, 1, width)
                pattern_array = interp1d(x_old, pattern_array, kind='linear')(x_new)
            control_points = pattern_array
        else:
            from scipy.interpolate import BSpline
            degree = 3
            cp = np.array(pattern['control_points'])
            n_cp = len(cp)
            knots = np.concatenate([np.zeros(degree + 1), np.linspace(0, 1, n_cp - degree + 1)[1:-1], np.ones(degree + 1)])
            control_points = BSpline(knots, cp, degree)(np.linspace(0, 1, width))
        
        if not use_relative:
            control_points = control_points * (data_max - data_min) + data_min
        
        train_feat = pattern_to_features(input_series_train_stacked, width, start, 
                                        pattern['series_idx'], pattern=control_points, data_min=data_min, data_max=data_max, 
                                        use_relative=use_relative, shift_tolerance=shift_tolerance)
        test_feat = pattern_to_features(input_series_test_stacked, width, start, 
                                       pattern['series_idx'], pattern=control_points, data_min=data_min, data_max=data_max,
                                       use_relative=use_relative, shift_tolerance=shift_tolerance)
        train_features.append(train_feat.reshape(-1, 1))
        test_features.append(test_feat.reshape(-1, 1))
    
    X_train, X_test = np.column_stack(train_features), np.column_stack(test_features)
    model = LightGBMModelWrapper('classification', n_classes=2)
    model.fit(X_train, y_train.values, X_test, y_val.values)
    
    # Find a representative ARDS case (or any case if no ARDS cases)
    ards_cases = np.where(y_val.values == 1)[0]
    correct_ards = ards_cases[0]
    
    explainer = shap.TreeExplainer(model.model)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    if len(shap_vals.shape) == 3: shap_vals = shap_vals[:, :, -1]
    
    feature_names = ['Age'] + [
        f"P{i+1}: {series_names[fold_1_patterns[i]['series_idx']]} "
        f"({'rel' if fold_1_patterns[i].get('use_relative', False) else 'abs'}, {fold_1_patterns[i].get('transform_type', 'raw')})"
        for i in range(len(fold_1_patterns))
    ]
    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    
    return shap.Explanation(values=shap_vals[correct_ards], base_values=base_value, data=X_test[correct_ards], feature_names=feature_names)

def process_azt1d(subject_id='1'):
    X_list, y, cgm0, series_names = load_azt1d_data(subject_id)
    
    # Try to load actual pattern parameters, fall back to representative example
    with open(f'../json_files/azt1d/pattern_parameters_{subject_id}.json', 'r') as f:
        pattern_data = json.load(f)
    if isinstance(pattern_data, list):
        patterns = pattern_data
    elif isinstance(pattern_data, dict) and 'patterns' in pattern_data:
        patterns = pattern_data['patterns']
    elif isinstance(pattern_data, dict) and 'fold_1' in pattern_data:
        patterns = pattern_data['fold_1']
    else:
        patterns = pattern_data
    
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42)
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    input_series_train = [x.iloc[train_idx] for x in X_list]
    input_series_test = [x.iloc[test_idx] for x in X_list]
    
    train_init = cgm0[train_idx].reshape(-1, 1)
    test_init = cgm0[test_idx].reshape(-1, 1)
    
    input_series_train_stacked = np.stack([x.values for x in input_series_train], axis=1)
    input_series_test_stacked = np.stack([x.values for x in input_series_test], axis=1)
    
    train_features, test_features = [train_init], [test_init]
    
    for pattern in patterns:
        data_min, data_max = np.min(input_series_train_stacked), np.max(input_series_train_stacked)
        width = int(pattern.get('width', 10))
        use_relative = pattern.get('use_relative', False)
        shift_tolerance = pattern.get('shift_tolerance', 0.0)
        start = pattern.get('start', int(pattern.get('center', 0) - width/2))
        
        # Create pattern array from pattern values or control points
        if 'pattern' in pattern and len(pattern['pattern']) > 0:
            pattern_array = np.array(pattern['pattern'])
            if len(pattern_array) != width:
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(pattern_array))
                x_new = np.linspace(0, 1, width)
                pattern_array = interp1d(x_old, pattern_array, kind='linear')(x_new)
            control_points = pattern_array
        else:
            from scipy.interpolate import BSpline
            degree = 3
            cp = np.array(pattern['control_points'])
            n_cp = len(cp)
            knots = np.concatenate([np.zeros(degree + 1), np.linspace(0, 1, n_cp - degree + 1)[1:-1], np.ones(degree + 1)])
            control_points = BSpline(knots, cp, degree)(np.linspace(0, 1, width))
        
        if not use_relative:
            control_points = control_points * (data_max - data_min) + data_min
        
        train_feat = pattern_to_features(input_series_train_stacked, width, start, 
                                        pattern['series_idx'], pattern=control_points, data_min=data_min, data_max=data_max,
                                        use_relative=use_relative, shift_tolerance=shift_tolerance)
        test_feat = pattern_to_features(input_series_test_stacked, width, start, 
                                       pattern['series_idx'], pattern=control_points, data_min=data_min, data_max=data_max,
                                       use_relative=use_relative, shift_tolerance=shift_tolerance)
        train_features.append(train_feat.reshape(-1, 1))
        test_features.append(test_feat.reshape(-1, 1))
    
    X_train, X_test = np.column_stack(train_features), np.column_stack(test_features)
    model = LightGBMModelWrapper('regression')
    model.fit(X_train, y_train.values, X_test, y_test.values)
    
    preds = model.predict(X_test)
    errors = np.abs(preds - y_test.values)
    best_pred = np.argmin(errors)
    
    explainer = shap.TreeExplainer(model.model)
    shap_vals = explainer.shap_values(X_test)
    if len(shap_vals.shape) == 3: shap_vals = shap_vals[:, :, -1]
    
    feature_names = ['CGM_0'] + [
        f"P{i+1}: {series_names[patterns[i]['series_idx']]} "
        f"({'rel' if patterns[i].get('use_relative', False) else 'abs'}, {patterns[i].get('transform_type', 'raw')})"
        for i in range(len(patterns))
    ]
    
    return shap.Explanation(values=shap_vals[best_pred], base_values=explainer.expected_value, data=X_test[best_pred], feature_names=feature_names)

def plot_custom_shap(explanation, ax, title, task_type='classification'):
    values = explanation.values
    features = explanation.feature_names
    base_value = explanation.base_values
    
    top_k = 8
    indices = np.argsort(np.abs(values))[::-1][:top_k]
    
    top_values = values[indices]
    top_features = [features[i] for i in indices]
    
    cumulative = [base_value]
    for v in top_values:
        cumulative.append(cumulative[-1] + v)
    
    final_pred = cumulative[-1]
    
    x_pos = np.arange(len(top_features))
    labels = top_features
    
    bar_starts = []
    bar_heights = []
    colors = []
    
    for i, v in enumerate(top_values):
        if v >= 0:
            bar_starts.append(cumulative[i])
            bar_heights.append(v)
            colors.append('#ff6b6b')
        else:
            bar_starts.append(cumulative[i+1])
            bar_heights.append(-v)
            colors.append('#4dabf7')
    
    bars = ax.bar(x_pos, bar_heights, bottom=bar_starts, color=colors, 
                   alpha=0.8, width=0.8)
    
    # Add horizontal lines connecting bars to show cumulative nature
    for i in range(len(x_pos)-1):
        ax.plot([x_pos[i]+0.4, x_pos[i+1]-0.4], 
                [cumulative[i+1], cumulative[i+1]], 
                'k-', linewidth=1.5, alpha=0.7)
    
    for i, (start, height, v_idx) in enumerate(zip(bar_starts, bar_heights, range(len(top_values)))):
        value = top_values[v_idx]
        y_text = start + height/2 if value >= 0 else start - height/2
        ax.text(x_pos[i], y_text, f'{value:.2f}', 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax.text(x_pos[-1] + 0.5, final_pred, f'  Final: {final_pred:.2f}', 
            va='center', ha='left', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#969696', alpha=0.3, edgecolor='black'))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha='center', fontsize=8)
    
    if task_type == 'classification':
        ax.set_ylabel('Log-odds (Model Output)', fontsize=11, fontweight='bold')
    else:
        ax.set_ylabel('Predicted Value (Model Output)', fontsize=11, fontweight='bold')
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    pos_patch = plt.Rectangle((0, 0), 1, 1, fc='#ff6b6b', alpha=0.8, edgecolor='black')
    neg_patch = plt.Rectangle((0, 0), 1, 1, fc='#4dabf7', alpha=0.8, edgecolor='black')
    base_line = plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.legend([pos_patch, neg_patch, base_line], 
             ['Positive contribution', 'Negative contribution', f'Base: {base_value:.2f}'],
             loc='upper left', fontsize=9, framealpha=0.9)

def main():
    
    # Process MIMIC-IV
    mimic_exp = process_mimic()
    
    # Process AZT1D
    azt1d_exp = process_azt1d(subject_id='1')
    shap_data = {
        'mimic': {
            'feature_names': mimic_exp.feature_names,
            'shap_values': mimic_exp.values.tolist(),
            'base_value': float(mimic_exp.base_values),
            'final_prediction': float(mimic_exp.base_values + mimic_exp.values.sum())
        },
        'azt1d': {
            'feature_names': azt1d_exp.feature_names,
            'shap_values': azt1d_exp.values.tolist(),
            'base_value': float(azt1d_exp.base_values),
            'final_prediction': float(azt1d_exp.base_values + azt1d_exp.values.sum())
        }
    }
    
    with open('../manuscript/tables/shap_values.json', 'w') as f:
        json.dump(shap_data, f, indent=2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    plot_custom_shap(mimic_exp, ax1, '(a) MIMIC-IV: ARDS Classification', task_type='classification')
    plot_custom_shap(azt1d_exp, ax2, '(b) AZT1D: Glucose Forecasting', task_type='regression')
    plt.tight_layout()
    plt.savefig('../manuscript/images/shap_waterfall.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()

