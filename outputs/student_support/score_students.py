"""Generate a teacher-reviewed queue from a frozen, unlabeled pre-exam CSV."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib

champion = None
KNOWN_CATEGORIES = {}
NUMERIC_BOUNDS = {}
RULES = {}

def logit_feature(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)

def predict_bundle(bundle, frame):
    p = bundle['pipeline'].predict_proba(frame[bundle['features']])[:, 1]
    return bundle['calibrator'].predict_proba(logit_feature(p))[:, 1]

def validate_records(records, features):
    required = ['student_id'] + features
    missing_columns = sorted(set(required) - set(records.columns))
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')
    if records.empty:
        raise ValueError('The scoring cohort is empty.')
    if records.student_id.isna().any() or records.student_id.astype(str).str.strip().eq('').any():
        raise ValueError('Every student needs a nonempty identifier.')
    if records.student_id.astype(str).duplicated().any():
        raise ValueError('Student identifiers must be unique within the scoring cohort.')
    clean = records.copy()
    clean['student_id'] = clean.student_id.astype(str)
    for c in features:
        if c in NUMERIC_BOUNDS:
            converted = pd.to_numeric(clean[c], errors='coerce')
            if (clean[c].notna() & converted.isna()).any():
                raise ValueError(f'{c} contains a nonnumeric value.')
            low, high = NUMERIC_BOUNDS[c]
            if (converted.notna() & ~converted.between(low, high)).any():
                raise ValueError(f'{c} must be within {low} to {high}.')
            if c == 'practice_tests_completed' and (converted.dropna() % 1 != 0).any():
                raise ValueError('Practice-test counts must be whole numbers.')
            clean[c] = converted
        elif c in KNOWN_CATEGORIES:
            unknown = set(clean[c].dropna()) - set(KNOWN_CATEGORIES[c])
            if unknown:
                raise ValueError(f'{c}: review unknown categories {sorted(map(str, unknown))}.')
    if clean[features].isna().all(axis=1).any():
        raise ValueError('At least one row has no measured predictors; use teacher review.')
    if 'internet_access' in clean:
        if not clean.internet_access.dropna().isin([0, 1]).all():
            raise ValueError('Internet access must be 0, 1, or missing.')
    if 'device_availability' in clean:
        if not clean.device_availability.dropna().isin(['Shared', 'Dedicated']).all():
            raise ValueError('Review unknown device availability values.')
    return clean

def recorded_support_needs(row):
    needs = []
    if pd.notna(row.get('attendance_percentage')) and row.attendance_percentage < RULES['attendance_below']:
        needs.append('Ask about attendance barriers')
    if pd.notna(row.get('assignment_completion_rate')) and row.assignment_completion_rate < RULES['assignment_below']:
        needs.append('Review unfinished assignments')
    if pd.notna(row.get('practice_tests_completed')) and row.practice_tests_completed < RULES['practice_below']:
        needs.append('Check readiness with a short practice assessment')
    if row.get('study_consistency') == 'Low' or row.get('revision_frequency') == 'Rarely':
        needs.append('Agree on a manageable revision plan')
    return '; '.join(needs) or 'Teacher checks subject-specific learning gaps'

def accessible_route(row):
    if row.get('internet_access') == 0 or row.get('device_availability') == 'Shared':
        return 'Ask whether printed material or in-person support is needed'
    if pd.isna(row.get('internet_access')) or pd.isna(row.get('device_availability')):
        return 'Confirm internet and device access with student'
    return 'Ask student to choose in-person or digital support'

def make_support_queue(records, slots, bundle=None):
    if bundle is None:
        bundle = champion
    if bundle is None:
        raise ValueError("Load a trusted model bundle before scoring.")
    if isinstance(slots, bool) or not isinstance(slots, (int, np.integer)) or not 0 <= slots <= len(records):
        raise ValueError('Slots must be a whole number from zero to the cohort size.')
    clean = validate_records(records, bundle['features'])
    probabilities = predict_bundle(bundle, clean)
    result = pd.DataFrame({'student_id': clean.student_id.to_numpy(),
        'estimated_failure_probability': probabilities,
        'missing_predictor_count': clean[bundle['features']].isna().sum(axis=1).to_numpy(),
        'recorded_support_needs': clean.apply(recorded_support_needs, axis=1).to_numpy(),
        'accessible_support_route': clean.apply(accessible_route, axis=1).to_numpy()})
    result = result.sort_values(['estimated_failure_probability', 'student_id'],
                                 ascending=[False, True]).reset_index(drop=True)
    result.insert(0, 'priority_rank', np.arange(1, len(result) + 1))
    result['offered_slot'] = result.priority_rank <= slots
    result['review_status'] = 'Awaiting teacher review'
    return result


def load_model(model_path):
    global champion, KNOWN_CATEGORIES, NUMERIC_BOUNDS, RULES
    artifact = joblib.load(model_path)
    champion = artifact['bundle']
    KNOWN_CATEGORIES = artifact['known_categories']
    NUMERIC_BOUNDS = artifact['numeric_bounds']
    RULES = artifact['rules']
    return artifact

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--slots', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=Path(__file__).with_name('student_support_model.joblib'))
    args = parser.parse_args()
    artifact = load_model(args.model)
    records = pd.read_csv(args.input)
    queue = make_support_queue(records, args.slots)
    if args.output.exists():
        raise FileExistsError('Choose a new output path to preserve the existing worklist.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False)
    print(json.dumps({'records': len(queue), 'offered_slots': int(queue.offered_slot.sum()),
                      'model': artifact['summary']['selected_model'],
                      'status': 'Awaiting teacher review'}, indent=2))

if __name__ == '__main__':
    main()
