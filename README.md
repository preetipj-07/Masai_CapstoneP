## Part 1: Dataset & Baseline Models

### Dataset Generation
- Script: `generate_orders.py`  
- Output: `orders_dataset.csv` with **6,000 rows and 13 columns**  
- Properties:  
  - Overall return-rate: **22.75%** (within 18–27% target range)  
  - Missing ratings: **13.05%** (within 8–18% target range)  
- Missingness classification: **MAR**  
  - COD orders have ~22.8% missing ratings vs non‑COD ~6.1%  
  - Evidence shows dependency on `payment_method` [not MCAR(Missing Completely At Random), not MNAR(Missing Not At Random)]

### Dataset Verification
- Script: `verify_dataset.py`  
- Confirmed dataset shape, return rate, and missing ratings %  
- Subgroup analysis:  
  - **Product category**: Apparel & Footwear highest (~26%), Electronics lowest (~18.7%)  
  - **Payment method**: COD highest (~30.7%), prepaid methods ~17%  
- Conclusion: Dataset does meet all acceptance criteria

###  Baseline Model
- Script: `baseline_model.py`  
- Model: DummyClassifier (most frequent strategy)  
- Results:  
  - Accuracy: **77.25%**  
  - F1 (returned=1): **0.0**  
- Explanation: High accuracy is misleading because the model always predicts “no return,” giving **zero recall** for returned orders.

###  Logistic Regression
- Script: `logistic_regression.py`  
- Model: Logistic Regression (`class_weight="balanced"`)  
- Default threshold (0.5):  
  - Accuracy: **0.5925**  
  - F1 (returned=1): **0.3925**  
  - Recall: **0.5788**  
  - Precision: **0.297**  
  - ROC‑AUC: **0.625**  
- Threshold sweep:  
  - Best threshold: **0.44**  
  - Max F1: **0.409**  
  - Recall: **0.758** (+18 percentage points vs default)  
  - Precision: **0.28** (drops as recall rises)  
- Business trade‑off: Lowering the threshold increases recall (catching more risky orders), but reduces precision (more false alarms). In support workflows, recall is prioritized because missing a true return is more costly than flagging a few safe orders.
