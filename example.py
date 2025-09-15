import xgboost as xgb
from scripts import RetentionModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PurityThresholdPruning import PurityThresholdPruning

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train a Random Forest model
# For PurityThresholdPruning, the model MUST have oob_score=True
# and X_train/y_train MUST be NumPy arrays
rf_model = RandomForestClassifier(n_estimators=100, oob_score=True)
rf_model.fit(X_train, y_train)
print("Accuracy Before Pruning:", rf_model.score(X_test, y_test))
rf_model = PurityThresholdPruning(rf_model, 0.05, X_train, y_train)
print("Accuracy After Pruning", rf_model.score(X_test, y_test))

rf_odr = RetentionModel(model=rf_model, model_type='RF', cam_size=64, strategy='ODR')
rf_spc = RetentionModel(model=rf_model, model_type='RF', cam_size=64, strategy='SPC')
rf_num = RetentionModel(model=rf_model, model_type='RF', cam_size=64, strategy='NUM')
rf_nim = RetentionModel(model=rf_model, model_type='RF', cam_size=64, strategy='NIM')
rf_odr.run_retention()
rf_spc.run_retention()
rf_num.run_retention()
rf_nim.run_retention()

# Train an XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

xgb_odr = RetentionModel(model=xgb_model, model_type='XGB', cam_size=64, strategy='ODR')
xgb_spc = RetentionModel(model=xgb_model, model_type='XGB', cam_size=64, strategy='SPC')
xgb_num = RetentionModel(model=xgb_model, model_type='XGB', cam_size=64, strategy='NUM')
xgb_nim = RetentionModel(model=xgb_model, model_type='XGB', cam_size=64, strategy='NIM')
xgb_odr.run_retention()
xgb_spc.run_retention()
xgb_num.run_retention()
xgb_nim.run_retention()