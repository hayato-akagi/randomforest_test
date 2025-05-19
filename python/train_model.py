# train_model.py
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("📊 Loading input.csv ...", flush=True)
df = pd.read_csv('/data/input.csv')

X = df[['magnetic_field', 'electric_field', 'position', 'momentum']]
y = df['label']

print("🧪 Splitting dataset ...", flush=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⚖️ Scaling features with StandardScaler ...", flush=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler parameters
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}
with open('/data/scaler.json', 'w') as f:
    json.dump(scaler_params, f)
print("💾 Scaler parameters saved to scaler.json", flush=True)

print("🌲 Training RandomForest ...", flush=True)
clf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)
clf.fit(X_train_scaled, y_train)

print("💾 Exporting model to rf_model.json ...", flush=True)
model_json = []
for estimator in clf.estimators_:
    tree = estimator.tree_
    model_json.append({
        'children_left': tree.children_left.tolist(),
        'children_right': tree.children_right.tolist(),
        'feature': tree.feature.tolist(),
        'threshold': tree.threshold.tolist(),
        'value': tree.value[:, 0, :].tolist(),
    })

with open('/data/rf_model.json', 'w') as f:
    json.dump(model_json, f)
    
print("✅ RandomForest model saved to /data/rf_model.json", flush=True)

# Predict test labels
y_pred = clf.predict(X_test_scaled)

print("📁 Saving test set to test.csv ...", flush=True)
test_df = X_test.copy()
test_df['label'] = y_test                # Ground truth
test_df['python_label'] = y_pred        # Model prediction
test_df.to_csv('/data/test.csv', index=False)

print("✅ train_model.py completed successfully!", flush=True)
