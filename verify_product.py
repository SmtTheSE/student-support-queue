from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import nbformat
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parent
out = root / 'outputs' / 'student_support'
spec = importlib.util.spec_from_file_location('student_scorer', out / 'score_students.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
artifact = scorer.load_model(out / 'student_support_model.joblib')
raw = pd.read_csv(root / 'student_exam_performance.csv')
records = raw.iloc[:40].copy()
queue = scorer.make_support_queue(records, 7)
assert len(queue) == 40 and queue.offered_slot.sum() == 7
assert queue.student_id.is_unique

# Labels and excluded demographic fields cannot change scoring.
changed = records.copy()
for c in ['exam_score', 'questions_correct', 'questions_attempted']:
    changed[c] = 0
for c in ['performance_grade', 'pass_status', 'performance_level', 'gender', 'family_income']:
    changed[c] = 'ALTERED'
pd.testing.assert_frame_equal(queue, scorer.make_support_queue(changed, 7))
pd.testing.assert_frame_equal(queue, scorer.make_support_queue(records.sample(frac=1, random_state=4), 7))
assert scorer.make_support_queue(records, 0).offered_slot.sum() == 0
assert scorer.make_support_queue(records, len(records)).offered_slot.all()

def must_reject(frame, slots=7):
    try:
        scorer.make_support_queue(frame, slots)
    except ValueError:
        return
    raise AssertionError('Invalid scoring input was accepted')

duplicate = records.copy()
duplicate.loc[duplicate.index[1], 'student_id'] = duplicate.iloc[0].student_id
must_reject(duplicate)
must_reject(records, -1)
must_reject(records, 2.5)
must_reject(records, True)
must_reject(records, len(records) + 1)
must_reject(records.iloc[:0], 0)
must_reject(records.drop(columns=[artifact['bundle']['features'][0]]))
bad_numeric = records.copy()
bad_numeric.loc[bad_numeric.index[0], 'attendance_percentage'] = 101
must_reject(bad_numeric)
bad_category = records.copy()
bad_category.loc[bad_category.index[0], 'study_consistency'] = 'Unseen category'
if 'study_consistency' in artifact['bundle']['features']:
    must_reject(bad_category)
missing = records.copy()
missing.loc[missing.index[0], artifact['bundle']['features']] = np.nan
must_reject(missing)
partial = records.copy()
partial.loc[partial.index[0], 'attendance_percentage'] = np.nan
partial_queue = scorer.make_support_queue(partial, 7)
assert partial_queue.estimated_failure_probability.between(0, 1).all()
assert partial_queue.loc[partial_queue.student_id.eq(partial.iloc[0].student_id), 'missing_predictor_count'].iloc[0] >= 1

nb = nbformat.read(out / 'student_support_queue.ipynb', as_version=4)
nbformat.validate(nb)
code_cells = [c for c in nb.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code_cells)
assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs)
all_outputs = '\n'.join(str(c.outputs) for c in code_cells)
assert 'STU_000001' not in all_outputs, 'Individual IDs leaked into notebook display'
summary = json.loads((out / 'results_summary.json').read_text())
assert summary['rows'] == len(raw)
assert summary['recorded_failures'] == int(raw.exam_score.lt(50).sum())
assert abs(summary['heldout_selected_policy']['failures_in_queue'] / summary['heldout_selected_policy']['slots']
           - summary['heldout_selected_policy']['precision_at_capacity']) < 1e-12

# Reconcile the saved-model product to the notebook's independent heldout evaluation.
labels = raw.exam_score.lt(50).astype(int)
_, remaining = train_test_split(raw.index.to_numpy(), test_size=.4, stratify=labels, random_state=42)
_, remaining = train_test_split(remaining, train_size=.25, stratify=labels.loc[remaining], random_state=43)
_, heldout = train_test_split(remaining, train_size=1/3, stratify=labels.loc[remaining], random_state=44)
heldout_queue = scorer.make_support_queue(raw.loc[heldout], 2000)
offered_ids = heldout_queue.loc[heldout_queue.offered_slot, 'student_id']
reached = int(raw.loc[raw.student_id.isin(offered_ids), 'exam_score'].lt(50).sum())
assert reached == summary['heldout_selected_policy']['failures_in_queue']

# Exercise real CSV input/output and refusal to overwrite a worklist.
with tempfile.TemporaryDirectory(prefix='student-support-check-') as directory:
    directory = Path(directory)
    input_path, output_path = directory / 'input.csv', directory / 'worklist.csv'
    records.drop(columns=['exam_score', 'performance_grade', 'pass_status', 'performance_level',
                          'questions_correct', 'questions_attempted']).to_csv(input_path, index=False)
    command = [sys.executable, str(out / 'score_students.py'), '--input', str(input_path),
               '--slots', '7', '--output', str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    saved = pd.read_csv(output_path)
    assert len(saved) == 40 and saved.offered_slot.sum() == 7
    blocked = subprocess.run(command, capture_output=True, text=True)
    assert blocked.returncode != 0 and 'Choose a new output path' in blocked.stderr

report = {'notebook': 'All code cells executed without errors',
          'product': 'Ranking invariance, outcome exclusion, slot limits, missing-data handling, and invalid-input rejection passed',
          'summary': 'Key counts and precision reconciled; saved model reproduces heldout queue reach',
          'csv_workflow': 'Unlabeled CSV scoring and output-preservation checks passed',
          'individual_notebook_output': 'No example student identifiers displayed'}
(out / 'verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
