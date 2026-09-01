# Machine Learning Assignments

## 📚 About

This repository contains my Machine Learning coursework — assignment scripts, notebooks, and datasets. Core algorithms (regression, classification, clustering) are implemented from scratch with NumPy/Pandas, with scikit-learn and Keras used only where the assignment explicitly permits it.

## 📁 Repository Structure

```
Machine_Learning_Assignment/
│
├── Assignments/
│   ├── Data/                          # Datasets used across the Scripts assignments
│   │   ├── car_performance.csv
│   │   ├── customer_segments.csv
│   │   ├── employee_data.csv
│   │   ├── employee_data_transformed.csv
│   │   ├── heart_disease.csv
│   │   ├── housing_price.csv
│   │   ├── loan_approval.csv
│   │   ├── synthetic_moons.csv
│   │   └── make_datasets.py           # Script that generates the synthetic datasets above
│   │
│   └── Scripts/                       # One folder per question set (A–H)
│       ├── A_data_preprocessing/
│       ├── B_simple_linear_regression/
│       ├── C_multiple_linear_regression/
│       ├── D_logistic_regression/
│       ├── E_knn/
│       ├── F_svm/
│       ├── G_decision_tree/
│       └── H_cnn_mnist/
│
├── Heart Failure Predection Model/
│   └── Heart_Failure_Prediction.ipynb  # End-to-end classification notebook
│
├── Normal_Equation/
│   ├── Normal_Equation.ipynb
│   ├── behavioral_indicators_internal_marks.csv
│   └── Readme.md                       # Assignment-specific notes
│
├── LICENSE                             # MIT License
└── README.md                           # This file
```



## 🛠 Technologies Used

- Python 3.13+
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras (assignment H only)
- kagglehub (Heart Failure notebook only)



## 📦 Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow kagglehub jupyter
```

*(No `requirements.txt` is included yet — consider adding one pinned to the versions you used.)*

## 📄 License

Released under the [MIT License](LICENSE) — © 2026 Nabin.