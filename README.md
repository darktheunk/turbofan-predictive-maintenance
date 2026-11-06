# Predictive Maintenance of Turbofan Engines Through Multivariate Sensor Data Analysis

Predicting the **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor time series, using the NASA C-MAPSS **FD001** subset.

**Status:** data understanding, EDA, preprocessing, feature engineering and model development (Phases 1 to 6) are complete. The official test-set evaluation, explainability and final report are still to do. The test set has not been used for any modelling decision.

---

## 1. Problem

RUL is the number of operating cycles an engine has left before it fails. Predicting it lets maintenance be planned before a failure, which reduces unexpected downtime and cost.

Given the sensor readings of an engine up to some cycle, predict how many cycles remain.

## 2. Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation), subset **FD001**, from the NASA Prognostics Center of Excellence data repository. Reference: A. Saxena and K. Goebel, "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository, 2008.

| Property | FD001 |
|---|---|
| Training engines (run to failure) | 100 |
| Test engines (cut off before failure) | 100 |
| Operating conditions | 1 |
| Fault modes | 1 |
| Training rows | 20,631 |
| Test rows | 13,096 |

Files in `data/raw/` (never modified, set read-only): `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`.

Each row is one engine at one cycle with 26 columns: `unit`, `cycle`, `setting_1..3`, `sensor_1..21`. The files have no header and are whitespace-separated. `RUL_FD001.txt` holds the true RUL at each test engine's last observed cycle, one value per engine in engine order.

Training RUL is computed as `max cycle of the engine - current cycle`.

## 3. Repository structure

```text
turbofan-predictive-maintenance/
├── data/
│   ├── raw/                 original NASA files (unchanged)
│   └── processed/           train_with_rul.csv, train_engineered.csv, preprocessing_config.json
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb      (planned, not yet run)
├── models/
├── results/                 experiment logs, CV tables, ablation tables
├── requirements.txt
└── README.md
```

The notebooks find the project root by searching upward for a `data/raw/` folder, so they run from any folder inside the project.

## 4. Setup (Fedora Linux)

```bash
cd ~/turbofan-predictive-maintenance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name turbofan --display-name "Python (turbofan)"
jupyter lab
```

Put the three FD001 files in `data/raw/`, then open the notebooks in order with the **Python (turbofan)** kernel. XGBoost was run on an NVIDIA GPU (`device="cuda"`). On a machine without a CUDA GPU, use `n_jobs=-1` instead. Results differ very slightly between CPU and GPU.

Libraries: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, XGBoost, JupyterLab.

## 5. Method

### 5.1 Data understanding and quality (notebooks 01 and 03)
- All 26 columns loaded with correct names; shapes match the expected 20,631 x 26 and 13,096 x 26.
- No missing values, duplicate rows, duplicate (unit, cycle) pairs or non-finite values, in train or test.
- Every engine's cycles run 1..N with no gaps, and rows are in time order. No data was deleted or filled in the raw data.
- Training engine lifetimes: 128 to 362 cycles, mean 206.3, right-skewed. Test observed lengths: 31 to 303. Test RUL: 7 to 145.

### 5.2 Exploratory analysis (notebook 02)
- Seven columns are constant in training and carry no information: `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`.
- `setting_1`, `setting_2` and `sensor_6` show no degradation trend.
- Sensors 2, 3, 4, 8, 11, 13, 15, 17 rise and sensors 7, 12, 20, 21 fall toward failure, in the same direction for all 100 engines.
- `sensor_9` and `sensor_14` respond strongly in some engines and barely in others, so their linear correlation with RUL is weak. They were kept, and dropping them later made results worse.
- Aligned by RUL, engines follow the same curve at different speeds. Sensors are flat at high RUL and only bend in roughly the last 100 to 150 cycles.
- 14 sensors kept: 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21.

### 5.3 Feature engineering (notebook 04, extended in notebook 05)
All features are computed per engine and look only backward (current and earlier cycles), checked by recomputing on truncated engines.
- Rolling mean and rolling standard deviation (window 10).
- Trailing least-squares slope of each sensor over a window (tested at 10, 20, 30, 40, 50, 60, 80, 100, 120 and the whole history).
- `cycle`, the engine's current age.
- The few NaNs (cycle 1 only, verified) were filled with 0.

### 5.4 Evaluation protocol
- **Split by engine ID, never by row.** An 80/20 engine split (`random_state=42`, IDs saved in `preprocessing_config.json`), then 5-fold **GroupKFold** over all 100 engines for model and feature comparisons.
- Transforms such as scaling are fit on training engines only.
- Models are scored against the **true, unclipped RUL**, on rows with RUL <= 145 to match the official test label range.
- Metrics: MAE, RMSE and the NASA asymmetric score (over-predicting remaining life is penalised more than under-predicting). The NASA score is a sum, so it is only compared on the same set of rows.

## 6. Results so far

All numbers below are from validation (grouped CV or the 20-engine validation split). **None come from the official test set.**

### 6.1 Baseline models (single split, unclipped, rows with RUL <= 145)

| Model | MAE | RMSE |
|---|---|---|
| Dummy (mean RUL) | 45.37 | 55.42 |
| Linear Regression | 25.00 | 30.07 |
| Random Forest | 17.14 | 24.93 |

Linear Regression produced curved errors, 208 negative predictions out of 4,070, and over-predicted life in the 25 to 125 cycle range.

### 6.2 RUL clipping (Random Forest, 5-fold grouped CV, mean MAE)

| Cap on training RUL | MAE |
|---|---|
| none | 20.32 |
| 150 | 15.24 |
| **130** | **13.99** |
| 110 | 14.42 |

A cap of 130 was chosen from these four tested values. It is justified by the flat sensors at high RUL, the test RUL never exceeding 145, and the CV result. Validation is always scored against the unclipped RUL. A capped model cannot predict above the cap.

### 6.3 Model comparison (cap 130, 5-fold grouped CV)

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression | 18.36 | 22.43 |
| Random Forest | 13.99 | 18.82 |
| XGBoost | **13.53** | **18.22** |

Features at this stage: 56 (14 raw sensors, rolling mean, rolling std and slope with window 20).

### 6.4 Feature ablations (XGBoost, CV MAE)

| Features | MAE |
|---|---|
| All 56 | 13.53 |
| Without sensors 9 and 14 | 14.21 |
| Raw sensors only | 16.48 |
| Without slope | 16.65 |
| Without std | 13.48 |
| All 56 + cycle | 12.33 |
| Raw + slope (window 20) | 13.50 |
| Raw + rolling mean | 16.67 |
| Raw + slope (window 20) + cycle | 12.41 |

Slope is the feature that matters, rolling mean and std add nothing for tree models, and `cycle` helps.

### 6.5 Slope window search (raw sensors + slope + cycle, CV MAE)

| Window | 10 | 20 | 30 | 40 | 50 | 60 | 80 | 100 | 120 | whole history |
|---|---|---|---|---|---|---|---|---|---|---|
| MAE | 14.49 | 12.41 | 10.70 | 9.45 | 8.75 | 8.19 | **7.84** | 8.02 | 8.19 | 7.81 |

The gain levels off at about 80, which was chosen as the smallest window matching the best. Combining several windows gave no further gain. Long windows help little in the first 50 cycles, and the 50 to 100 cycle range remains the weakest.

### 6.6 Frozen configuration

| Item | Choice |
|---|---|
| Features (29) | 14 raw sensors, slope (window 80), `cycle` |
| Training target | RUL capped at 130 |
| Model | XGBoost: 400 trees, learning rate 0.05, max depth 5, subsample 0.8, colsample 0.8, seed 42 |
| Grouped 5-fold CV MAE | **7.84** (std 0.53) |

Note: the window-80 slope features are built inside the notebook code and have not yet been saved to a processed file.

## 7. Limitations and caveats

- Many variants were compared on the same 100 engines, so the CV numbers are somewhat optimistic. The official test set is the unbiased check.
- It is not yet understood why long slope windows help so much. This is reported as an empirical result.
- `cycle` helps in CV. It is available at prediction time, but its benefit still has to be confirmed on the official test protocol.
- The rows with RUL <= 145 used for scoring come from the test labels' range.
- `sensor_6` and the operating settings were not ablated. Hyperparameters were not tuned. LSTM, GRU and 1D CNN have not been tried.
- FD001 has a single operating condition and fault mode, so results do not transfer to FD002 to FD004 without further work.

## 8. Remaining work

1. `06_model_evaluation.ipynb`: build the frozen features for the test engines with the same function used for training, train on all 100 training engines, and score **only the last cycle of each test engine** against `RUL_FD001.txt` (MAE, RMSE, NASA score, bias, error by observed length). Run once, report as it comes out, and do not tune on it afterwards.
2. Optional deep learning (LSTM, GRU, 1D CNN) compared against XGBoost.
3. SHAP and sensor-contribution analysis, actual-versus-predicted plots, a final model comparison table.
4. Final report and presentation. Possible extras: inference pipeline or API and a results dashboard.

## 9. Team

| Member | Role | Responsibility |
|---|---|---|
| _Name 1_ | Data Science | Dataset analysis, preprocessing, feature engineering, model development |
| _Name 2_ | Backend | Model integration, inference pipeline, API if required |
| _Name 3_ | Visualization | Results visualization, dashboard or interface, documentation and presentation |

All members should understand the problem, the dataset, RUL prediction and the final model.

## 10. Reproducibility

- Raw data is never modified. Derived data goes to `data/processed/`, outputs to `results/`.
- The engine split, kept columns and chosen settings are recorded in `data/processed/preprocessing_config.json`.
- Every comparison uses the same engine-grouped folds and fixed random seeds.
- Experiment tables are saved in `results/`.
