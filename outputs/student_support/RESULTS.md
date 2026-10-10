# Student Support Queue: Results

The supplied CSV contains 100,000 records and 44 columns. 22,611 records (22.6%) have exam scores below 50. This is a file-level result; source provenance is unverified.

The product is a teacher-reviewed pre-exam support queue. Its selected model is Study-record logistic. Current-exam answers and outcome labels are excluded from scoring, along with demographic and socioeconomic ranking inputs.

On 20,000 heldout records with 2,000 slots, the queue contains 1,353 failures. Precision is 67.7%, and recall is 29.9%. The administrative model reaches 1,162 failures with the same capacity. The difference is +191 failures reached. The queue leaves 3,169 failures outside its capacity.

Test ROC AUC is 0.818; average precision is 0.571. Test Brier score is 0.1329. Model choice and capacity were fixed using separate selection data before final evaluation.

The analysis measures reach, not prevented failures. Business scenarios require explicit participation, program-effect, and cost assumptions. No intervention impact is measured in this file.

Use the product only after confirming dataset origin, field timestamps, school/subject definitions, and later-exam performance. The data lack dates and school IDs, so the random holdout does not establish generalization to new schools or future exams.
