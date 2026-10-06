# Does TabPFN-3.5 matter when the context is high-cardinality? (written before any run)

Task: TabLint's numerical cell-error benchmark (H6), reusing `inject()` from
TabLint `benchmarks/proofread_confirm.py` exactly (3% of continuous cells; decimal, swap, offset, zero, transposition).
Score = TabLint's PIT surprise -log(2*min(F, 1-F)), 5-fold out-of-fold, TabPFNRegressor
`create_default_for_version(v, ignore_pretraining_limits=True)`, device cpu.

Datasets (chosen because a high-cardinality column drives the numeric columns):
- employee_salaries (OpenML 42125), n=1000 sampled rows. Corrupted: current_annual_salary, 2016_gross_pay_received,
  2016_overtime_pay, year_first_hired. Low-card context: gender, assignment_category.
  High-card context: employee_position_title (385), division (694), department (37).
- medical_charges (OpenML 42720), n=1000 sampled rows. Corrupted: Total_Discharges, Average_Covered_Charges,
  Average_Total_Payments, Average_Medicare_Payments. Low-card: none. High-card: DRG_Definition (100),
  Provider_State (51), Hospital_Referral_Region (306), Provider_Id (3337).
- auto_mpg (TabLint demo/video/auto_mpg.csv, all 398 rows). Corrupted: mpg, displacement, horsepower, weight,
  acceleration. Low-card: cylinders, model_year, origin. High-card: car (~305).

Arms:
- `num`: other corrupted columns + low-card columns (what TabLint gives TabPFN today: it drops >30-level text).
- `hc`: `num` + high-card columns as declared categorical features (ordinal codes from sorted unique values).
Models: TabPFN v3, TabPFN-3.5. Seeds 1001-1005 -> 15 tables per (model, arm).

Primary hypothesis H1: on `hc`, 3.5 precision@k > v3 (paired over 15 tables, one-sided Wilcoxon, alpha .05).
H2: (3.5 hc - 3.5 num) > (v3 hc - v3 num), paired over 15 tables.
Secondary, descriptive: AUROC, per-dataset means, seconds. Also report `num` arm 3.5 vs v3 (expected ~tie, as in TabLint).
All runs reported; no dataset or seed dropped.
