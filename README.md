# 🧠 RandomForest Prediction Comparison (Python vs Go)

This project compares prediction results between:

- A **RandomForestClassifier** trained in Python (`scikit-learn`)
- A **Go implementation** that loads the same model from a JSON file

It also evaluates the impact of data preprocessing using `StandardScaler`.

---

## 📁 Project Structure

```

.
├── go/
│   ├── go.mod                # Go module file
│   ├── go.sum                # Go dependencies
│   ├── Dockerfile            # Dockerfile for Go environment
│   ├── main.go               # Go code for model inference
│   ├── main/                 # (Optional) compiled Go binary
│   └── wait\_for\_test\_csv.sh # Wait script for test.csv readiness
│
├── python/
│   ├── Dockerfile            # Dockerfile for Python environment
│   ├── main.py               # Entry script: trains model + generates test set
│   ├── train_model.py        # Train RF + Scaler, export as JSON
│   ├── generate_dataset.py   # Create input/test datasets
│   └── requirements.txt      # Python dependencies
│
├── data/
│   ├── input.csv             # Generated input features
│   ├── test.csv              # Test set + `python_label`
│   ├── rf_model.json         # RandomForest model exported from Python
│   └── scaler.json           # Scaler parameters (mean, scale)
│
├── run.sh                    # Runs full pipeline (Python then Go)
├── docker-compose.yml        # Spins up both containers
├── .gitignore
└── README.md                 # You're here!

````

---

## 🚀 Quick Start

### 🐳 Option 1: Docker Compose (Recommended)

```bash
docker compose up --build
````

This will:

1. Build and run the Python container

   * Generate `input.csv`
   * Train and export RandomForest (`rf_model.json`)
   * Save `scaler.json`
   * Output `test.csv` with predictions (`python_label`)
2. Build and run the Go container

   * Waits for `test.csv`
   * Loads `rf_model.json`, optionally applies `scaler.json`
   * Makes predictions in Go (`go_label`, `fixed_go_label`)
   * Appends Go results to `test.csv`
   * Prints comparison summary

---

### 🐍 Option 2: Run Python Only

```bash
cd python
docker build -t rf-python .
docker run --rm -v $(pwd)/../data:/data rf-python
```

---

### 🦫 Option 3: Run Go Only (after generating Python model)

```bash
cd go
docker build -t rf-go .
docker run --rm -v $(pwd)/../data:/data rf-go
```

---

## 📤 Output Example

After running the full pipeline (data generation, training, and inference), the following output shows a comparison of predicted labels between Python and Go implementations.

| magnetic\_field | electric\_field | position | momentum | label | python\_label | go\_label | fixed\_go\_label |
| --------------- | --------------- | -------- | -------- | ----- | ------------- | --------- | ---------------- |
| -0.34           | 288.29          | 9.85     | -0.787   | 1     | 1             | 1         | 1                |
| -0.049          | 240.14          | 2.5      | -0.036   | 0     | 1             | 1         | 1                |

---

## ✅ Label Meaning

* `label`: Ground truth from dataset
* `python_label`: Prediction from Python (ground truth)
* `go_label`: Majority vote across trees in Go
* `fixed_go_label`: Average of all tree probabilities (more stable)

---

## 🧠 Theoretical Model Behind the Dataset

We can base our model on **spin dynamics in external fields**, particularly the **Pauli equation** and the **Zeeman effect**. The spin state of an electron (up or down) can be influenced by:

* Magnetic moment interaction with magnetic field (Zeeman effect)
* Electric field interactions (e.g., spin-orbit coupling)
* Kinematic properties (position & momentum)

---

### 🧲 1. Zeeman Effect (Magnetic Field Interaction)

The **Zeeman energy splitting** between spin-up and spin-down states in an external magnetic field $\vec{B}$ is given by:

$$
\Delta E = 2 \mu_B B
$$

where:

* $\mu_B = \frac{e \hbar}{2 m_e}$ is the Bohr magneton
* $B$ is the magnetic field strength
* The energy difference results in preference toward spin-up or spin-down states.

The **Hamiltonian term** for spin in a magnetic field is:

$$
\hat{H}_{\text{Zeeman}} = -\vec{\mu} \cdot \vec{B} = -g \mu_B \hat{\vec{S}} \cdot \vec{B}
$$

where $\hat{\vec{S}}$ is the spin operator.

---

### ⚡ 2. Spin-Orbit Coupling (Electric Field & Momentum)

Spin-orbit coupling can be modeled as:

$$
\hat{H}_{\text{SO}} \propto \vec{E} \cdot (\vec{p} \times \vec{\sigma})
$$

Where:

* $\vec{E}$: Electric field
* $\vec{p}$: Momentum
* $\vec{\sigma}$: Pauli matrices (spin)

This implies spin state can be influenced by electric fields, especially in systems like atoms, semiconductors, or quantum dots.

---

### 🧮 Simplified Model (Our Simulation)

In the simulation, we used a **logical heuristic rule** based on:

```python
label = 1 if magnetic_field * momentum > 0.05 and electric_field > 150 else 0
```

Which implicitly encodes:

* $B \cdot p > \text{threshold}$: Emulates Zeeman-like alignment between momentum and magnetic field
* $E > \text{threshold}$: Activates electric-field-based spin interaction (like spin-orbit coupling)

---

### 🧪 Summary of this model

| Physical Effect        | Modeled by                    | Real Equation                                                             |
| ---------------------- | ----------------------------- | ------------------------------------------------------------------------- |
| Zeeman Effect          | `magnetic_field * momentum`   | $\hat{H} = -g \mu_B \hat{\vec{S}} \cdot \vec{B}$                          |
| Spin-Orbit Coupling    | `electric_field` + `momentum` | $\hat{H}_{\text{SO}} \propto \vec{E} \cdot (\vec{p} \times \vec{\sigma})$ |
| Combined spin behavior | Learned by `RandomForest`     | From synthetic labels using combined thresholds                           |

---


## 📌 Notes

* Model and scaler are serialized in JSON for portability.
* `run.sh` can orchestrate full process locally if desired.
* Go predictions work independently of Python runtime once `rf_model.json` and `test.csv` exist.

---

## 🧑‍💻 License

MIT License
