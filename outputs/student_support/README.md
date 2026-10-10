# Student Support Queue

A data science product that helps teachers prioritize limited support slots before an exam.

Open `student_support_queue.ipynb` for the complete executed analysis. Read `RESULTS.md` for the findings. The notebook contains raw-data checks, feature timing rules, descriptive insights, model comparisons, heldout evaluation, uncertainty intervals, group audits, a scoring function, and explicit business scenarios.

## Kaggle

1. Create or attach a dataset containing `student_exam_performance.csv`. Confirm the source's usage and publication rights; no license was supplied with this file.
2. Create a Python notebook and import `student_support_queue.ipynb`.
3. Attach the dataset as input and run all cells on CPU. Internet can remain disabled.
4. Save a version after reviewing the results. Individual worklist export is disabled by default; the notebook displays aggregate evidence.

The loader finds the CSV under `/kaggle/input`. Set `DATA_PATH` only if several files have that name. No additional local scripts are required to run the notebook. See [Kaggle notebooks](https://www.kaggle.com/docs/notebooks) and [Kaggle datasets](https://www.kaggle.com/docs/datasets).

## Score a new pre-exam cohort locally

Prepare a CSV using the headers in `scoring_input_template.csv`. Keep actual missing measurements blank. Use only records frozen before the exam. Confirm study fields are available at the checkpoint; otherwise evaluate the administrative baseline rather than using the richer model.

From this folder, run:

```bash
python score_students.py --input new_pre_exam_snapshot.csv --slots 25 --output restricted_teacher_worklist.csv
```

The script loads `student_support_model.joblib`, validates the input, scores records without requiring exam outcomes, and writes a ranked worklist. It retains all students and marks which have an offered slot. It refuses to overwrite an existing output file. Use the notebook's recorded package versions for the saved model, or rerun the notebook to rebuild it in your runtime. Load only a trusted model artifact.

The worklist contains student identifiers and must remain within the school's authorized support workflow. Teachers confirm needs and choose support with the student. The score is not a grade or a diagnosis.

## Evidence files

- `RESULTS.md` and `results_summary.json`: computed findings and run metadata.
- `data_quality.csv` and `data_dictionary.csv`: raw-file checks and the role of all 44 fields.
- `model_selection.csv`: candidates compared on separate selection data.
- `heldout_policy_comparison.csv`, `heldout_probability_metrics.csv`, and `heldout_bootstrap_intervals.csv`: final-test evidence.
- `capacity_sensitivity.csv` and `group_audit.csv`: staffing tradeoffs and group-level errors.
- `business_scenarios.csv`: measured queue precision combined with clearly labeled cost and intervention assumptions.
- `feature_importance.csv` and PNG charts: supporting analysis.

## Claim boundaries

Source provenance and pre-exam collection times are unverified. The file lacks school IDs, exam dates, subjects, and intervention outcomes. Random holdout performance describes these records; it does not prove performance in later exams or real schools. No prevented failures, financial savings, or causal effects are claimed as observed results.

Use a prospective local pilot to verify the data contract, calibration, access to support, and actual exam outcomes. The notebook describes separate evaluations of support effectiveness and ranking effectiveness.
