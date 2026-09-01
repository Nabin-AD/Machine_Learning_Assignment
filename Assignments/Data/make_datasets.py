import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------- A: employee_data.csv ----------
n = 40
depts = rng.choice(["Sales","Engineering","HR","Marketing","Finance"], n)
age = rng.integers(22, 60, n).astype(float)
experience = np.clip(age - rng.integers(20,25,n), 0, None).astype(float)
salary = 25000 + experience*1800 + rng.normal(0,3000,n)
df_a = pd.DataFrame({
    "EmployeeID": np.arange(1, n+1),
    "Age": age,
    "Department": depts,
    "YearsExperience": experience,
    "Salary": salary.round(2)
})
# inject missing values
for col, frac in [("Age",0.1), ("YearsExperience",0.1), ("Salary",0.08), ("Department",0.08)]:
    idx = rng.choice(n, int(n*frac), replace=False)
    df_a.loc[idx, col] = np.nan
df_a.to_csv("/home/claude/lab/data/employee_data.csv", index=False)

# ---------- B: housing_price.csv ----------
n = 60
area = rng.uniform(500, 4000, n)
price = 50000 + area*120 + rng.normal(0, 25000, n)
df_b = pd.DataFrame({"FloorArea_sqft": area.round(1), "Price": price.round(2)})
df_b.to_csv("/home/claude/lab/data/housing_price.csv", index=False)

# ---------- C: car_performance.csv ----------
n = 80
engine = rng.uniform(1.0, 5.0, n)
hp = rng.uniform(70, 400, n)
price = 8000 + engine*3500 + hp*90 + rng.normal(0, 2500, n)
df_c = pd.DataFrame({"Engine_Size_L": engine.round(2), "Horsepower": hp.round(1), "Price_USD": price.round(2)})
df_c.to_csv("/home/claude/lab/data/car_performance.csv", index=False)

# ---------- D: heart_disease.csv ----------
n = 150
age = rng.integers(29, 77, n)
chol = rng.integers(150, 320, n)
bp = rng.integers(90, 200, n)
maxhr = rng.integers(80, 200, n)
logit = -6 + 0.05*age + 0.015*chol + 0.02*bp - 0.03*maxhr
prob = 1/(1+np.exp(-logit))
target = (rng.uniform(0,1,n) < prob).astype(int)
df_d = pd.DataFrame({"Age":age,"Cholesterol":chol,"BloodPressure":bp,"MaxHeartRate":maxhr,"Target":target})
df_d.to_csv("/home/claude/lab/data/heart_disease.csv", index=False)

# ---------- E: customer_segments.csv ----------
n = 120
centers = np.array([[20,20],[70,70],[20,80],[80,20]])
labels = rng.integers(0,4,n)
pts = centers[labels] + rng.normal(0,7,(n,2))
df_e = pd.DataFrame({"AnnualIncome_k": pts[:,0].round(1), "SpendingScore": pts[:,1].round(1), "Segment": labels})
df_e.to_csv("/home/claude/lab/data/customer_segments.csv", index=False)

# ---------- F: synthetic_moons.csv (moons via numpy, no sklearn) ----------
n = 200
n1 = n//2
theta1 = rng.uniform(0, np.pi, n1)
x1 = np.cos(theta1); y1 = np.sin(theta1)
theta2 = rng.uniform(0, np.pi, n - n1)
x2 = 1 - np.cos(theta2); y2 = 1 - np.sin(theta2) - 0.5
X = np.vstack([np.column_stack([x1,y1]), np.column_stack([x2,y2])])
X += rng.normal(0, 0.12, X.shape)
y = np.array([0]*n1 + [1]*(n-n1))
df_f = pd.DataFrame({"X1":X[:,0].round(4), "X2":X[:,1].round(4), "Label":y})
df_f.to_csv("/home/claude/lab/data/synthetic_moons.csv", index=False)

# ---------- G: loan_approval.csv ----------
n = 150
income = rng.uniform(2000, 15000, n)
loan_amt = rng.uniform(50000, 500000, n)
credit_score = rng.integers(300, 850, n)
dependents = rng.integers(0,4,n)
score = 0.001*income - 0.00002*loan_amt + 0.01*credit_score - 0.5*dependents
approved = (score > np.median(score)).astype(int)
df_g = pd.DataFrame({
    "Income": income.round(2), "LoanAmount": loan_amt.round(2),
    "CreditScore": credit_score, "Dependents": dependents, "Approved": approved
})
df_g.to_csv("/home/claude/lab/data/loan_approval.csv", index=False)

print("All datasets created:")
for f in ["employee_data","housing_price","car_performance","heart_disease","customer_segments","synthetic_moons","loan_approval"]:
    d = pd.read_csv(f"/home/claude/lab/data/{f}.csv")
    print(f, d.shape)
