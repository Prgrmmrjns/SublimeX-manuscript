import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def preprocess_mimic_data():
    """
    Preprocess MIMIC-IV data to create one row per subject with 24-hour time series features.
    """
    print("Loading MIMIC-IV data...")
    df = pd.read_csv('mimic_iv_binned_12.csv')
    
    print(f"Original data: {len(df)} rows, {df['subject_id'].nunique()} subjects")
    
    # Clean data
    df = df.dropna(subset=['subject_id', 'hour', 'ARDS_FLAG'])
    df['hour'] = df['hour'].astype(int)
    df = df.sort_values(['subject_id', 'hour'])
    
    # Clean extreme values and infinities
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    # Cap extreme values at 99.9th percentile to avoid scaling issues
    for col in numeric_cols:
        if col not in ['subject_id', 'hour', 'ARDS_FLAG']:
            q99 = df[col].quantile(0.999)
            q01 = df[col].quantile(0.001)
            df[col] = df[col].clip(lower=q01, upper=q99)
    
    # Define time series features with clean names
    series_mapping = {
        'Respiratory Rate [insp/min]': 'respiratory_rate',
        'Heart Rate [bpm]': 'heart_rate', 
        'O2 saturation pulseoxymetry [%]': 'o2_saturation',
        'Non Invasive Blood Pressure diastolic [mmHg]': 'bp_diastolic',
        'Non Invasive Blood Pressure systolic [mmHg]': 'bp_systolic',
        'Foley [mL]': 'foley',
        'Arterial Blood Pressure mean [mmHg]': 'bp_arterial_mean',
        'Temperature Fahrenheit [°F]': 'temperature',
        'GCS - Eye Opening [nan]': 'gcs_eye',
        'GCS - Verbal Response [nan]': 'gcs_verbal',
        'GCS - Motor Response [nan]': 'gcs_motor',
        'Inspired O2 Fraction [nan]': 'fio2',
        'PEEP set [cmH2O]': 'peep_set',
        'Norepinephrine [mg]': 'norepinephrine',
        'Creatinine (serum) [nan]': 'creatinine',
        'Platelet Count [nan]': 'platelets',
        'WBC [nan]': 'wbc',
        'PH (Arterial) [nan]': 'ph_arterial',
        'Phenylephrine [mg]': 'phenylephrine',
        'Arterial O2 pressure [mmHg]': 'pao2',
        'INR [nan]': 'inr',
        'Lactic Acid [nan]': 'lactate',
        'Total PEEP Level [cmH2O]': 'peep_total',
        'Total Bilirubin [nan]': 'bilirubin',
        'Dobutamine [mg]': 'dobutamine',
        'Epinephrine [mg]': 'epinephrine',
        'Dopamine [mg]': 'dopamine',
        'PaO2_FiO2': 'pao2_fio2',
        'Pulse Pressure [mmHg]': 'pulse_pressure'
    }
    
    # Filter to available series in the dataset
    available_series = {orig: clean for orig, clean in series_mapping.items() 
                       if orig in df.columns}
    
    print(f"Using {len(available_series)} time series features:")
    for orig, clean in available_series.items():
        print(f"  {orig} -> {clean}")
    
    # Create subject-level dataset
    processed_data = []
    
    for subj_id, group in df.groupby('subject_id'):
        group = group.sort_values('hour')
        
        # Skip subjects with too little data
        if len(group) < 5:
            continue
            
        # Get target (max ARDS_FLAG for this subject)
        target = int(group['ARDS_FLAG'].max())
        
        # Get anchor_age (should be same for all rows of this subject)
        anchor_age = group['anchor_age'].iloc[0]
        
        # Create row for this subject
        subject_row = {
            'subject_id': subj_id, 
            'anchor_age': anchor_age,
            'ARDS_FLAG': target
        }
        
        # Process each time series
        valid_subject = True
        for orig_name, clean_name in available_series.items():
            # Fill missing values within subject
            values = group[orig_name].fillna(method='ffill').fillna(method='bfill')
            
            if len(values) == 0 or values.isna().all():
                # If all NaN, fill with zeros for this series
                values = pd.Series([0.0] * len(group))
            
            # Create 24-hour sequence (pad or truncate)
            if len(values) >= 24:
                series_vals = values.iloc[:24].values
            else:
                # Pad with last valid value
                last_val = values.iloc[-1] if not pd.isna(values.iloc[-1]) else 0.0
                series_vals = np.concatenate([values.values, 
                                            np.full(24 - len(values), last_val)])
            
            # Add to subject row with hour indices
            for hour in range(24):
                col_name = f"{clean_name}_hour_{hour}"
                subject_row[col_name] = float(series_vals[hour])
        
        # Only add subjects with valid anchor_age
        if not pd.isna(anchor_age):
            processed_data.append(subject_row)
    
    # Create DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    print(f"Processed data: {len(processed_df)} subjects")
    print(f"Time series features: {len(available_series)}")
    print(f"Total features per subject: {len(processed_df.columns) - 3}")  # Exclude subject_id, anchor_age, target
    print(f"Class distribution: {processed_df['ARDS_FLAG'].value_counts().sort_index().to_dict()}")
    
    # Save processed data
    output_file = 'mimic_processed.csv'
    processed_df.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")
    
    return processed_df

if __name__ == "__main__":
    processed_df = preprocess_mimic_data()
    
    # Show sample of processed data
    print("\nSample of processed data:")
    print(processed_df[['subject_id', 'anchor_age', 'ARDS_FLAG', 
                       'respiratory_rate_hour_0', 'heart_rate_hour_0', 'o2_saturation_hour_0']].head())
    
    print(f"\nFirst 10 columns: {list(processed_df.columns[:10])}")
    print(f"Last 10 columns: {list(processed_df.columns[-10:])}")