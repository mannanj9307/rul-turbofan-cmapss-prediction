Problem Statement
Rolls-Royce Aerospace needs to predict the Remaining Useful Life (RUL) of turbofan engines from multi-sensor time-series data. Accurate predictions let maintenance be scheduled before a failure happens, which reduces unplanned downtime and improves efficiency.

Dataset
NASA C-MAPSS (Turbofan Engine Degradation Simulation), subset FD001
100 training engines (20,631 rows) and 100 test engines (13,096 rows)
3 operational settings and 21 sensor readings per cycle, with no missing values
Training engines run for 128 to 362 cycles before failure (mean about 206)
Source: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
The notebook downloads the dataset automatically with kagglehub.
Workflow
Load data -> EDA & visualization -> Pre-processing -> Model training -> Evaluation -> Visualization -> Maintenance logic
Data loading: read train_FD001, test_FD001 and RUL_FD001
EDA and visualization: lifetime distribution, sensor standard deviations, sensor degradation trends, correlation heatmap
Pre-processing
Compute the RUL label for each training cycle and cap it at 125 (piecewise-linear degradation)
Drop 8 constant columns (op2, op3, s1, s5, s10, s16, s18, s19), leaving 16 features
Scale features with MinMaxScaler (fit on training data only)
Build sliding windows of 30 cycles (window length = 30)
Engine-wise split (15 validation engines) to avoid data leakage
Test set: last 30-cycle window of each test engine
Shapes: Train (15104, 30, 16), Val (2627, 30, 16), Test (100, 30, 16)
Model building
Baseline: Random Forest (100 trees, on flattened windows)
Deep models: SimpleRNN, LSTM, GRU, BiLSTM (two recurrent layers with 64 and 32 units, Dropout 0.2, Dense 32, output Dense 1)
Adam (lr = 1e-3), MSE loss, batch size 128, up to 60 epochs, EarlyStopping (patience 8) and ReduceLROnPlateau
Evaluation: RMSE, MAE, R² and the NASA asymmetric scoring function
Visualization: model comparison, validation loss curves, predicted vs actual RUL, error distribution
Maintenance scheduling logic: predicted RUL is turned into maintenance priorities
Results (FD001 test set, 100 engines)
Model	RMSE	MAE	R²	NASA Score
Random Forest (baseline)	16.142	12.036	0.838	634.205
SimpleRNN	13.561	9.746	0.885	382.411
LSTM	13.353	9.859	0.889	324.896
GRU	13.815	9.979	0.881	374.394
BiLSTM	13.495	9.731	0.887	383.615
Lower RMSE, MAE and NASA score are better. Higher R² is better.

Best model: LSTM, with the lowest RMSE and NASA score and the highest R². Compared with the Random Forest baseline, it cut RMSE from 16.14 to 13.35 (about 17% lower) and the NASA score from 634 to 325 (about 49% lower). The four deep models perform close to each other, and BiLSTM has a slightly lower MAE.

Model comparison Validation loss Predicted vs Actual RUL RUL for all test engines Error distribution

Maintenance Scheduling
Engines are flagged for maintenance when predicted RUL is below 30 cycles. With the LSTM model, 24 of 100 engines were flagged.

Precision	Recall	F1
OK	0.97	0.99	0.98
Maintain	0.96	0.92	0.94
Overall accuracy was 0.97: 23 of 25 at-risk engines were caught, with 1 false alarm and 2 missed.

Predicted RUL is also converted into priority levels:

Priority	Predicted RUL (cycles)	Engines
CRITICAL	0 to 15	8
HIGH	15 to 30	16
MEDIUM	30 to 60	11
LOW	above 60	65
Exploratory Data Analysis
Lifetime distribution Sensor standard deviation Sensor trends Correlation heatmap

Key Findings
Sequence models (SimpleRNN, LSTM, GRU, BiLSTM) beat the Random Forest baseline because they learn temporal degradation patterns.
Capping RUL at 125 and dropping constant sensors improves training stability.
The LSTM gave the best overall accuracy and the lowest NASA score, which penalizes late predictions more heavily.
Predicted RUL converts into a maintenance schedule that catches 92% of engines that are close to failure.
How to Run
Open RUL_Turbofan_CMAPSS.ipynb in Google Colab and set the runtime to GPU (T4).
Run all cells (Runtime -> Run all). Training the four neural networks takes roughly 10 to 20 minutes.
Or locally:

bash
pip install -r requirements.txt
jupyter notebook RUL_Turbofan_CMAPSS.ipynb
Tech Stack
Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, kagglehub

Future Work
Other subsets (FD002 to FD004) with multiple operating conditions
Attention layers and Transformers
Deployment as a dashboard
Team
[Add your names and roll numbers]

Acknowledgement
Dataset: NASA Prognostics Center of Excellence, C-MAPSS turbofan engine degradation simulation.


Claude finished the response
