# Titanic Dataset Preprocessing

## Internship Task

**Organization:** ELEBATE LABS
**Internship:** AI/ML Internship
**Day:** 1
**Task:** 1 – Data Preprocessing
**Dataset:** Titanic Dataset

---

##  Objective

The main objective of this task is to understand and implement the fundamental steps involved in **data preprocessing** using the Titanic dataset.

Data preprocessing is an important step in Machine Learning because raw datasets often contain missing values, categorical variables, inconsistent data, and outliers. Proper preprocessing helps prepare the data for further analysis and machine learning model development.

---

## 📂 Dataset

The **Titanic dataset** contains information about passengers who were aboard the RMS Titanic.

Some important features include:

| Feature       | Description                          |
| ------------- | ------------------------------------ |
| `PassengerId` | Unique ID assigned to each passenger |
| `Survived`    | Survival status (0 = No, 1 = Yes)    |
| `Pclass`      | Passenger class (1st, 2nd, 3rd)      |
| `Name`        | Passenger name                       |
| `Sex`         | Gender of the passenger              |
| `Age`         | Age of the passenger                 |
| `SibSp`       | Number of siblings/spouses aboard    |
| `Parch`       | Number of parents/children aboard    |
| `Ticket`      | Ticket number                        |
| `Fare`        | Passenger fare                       |
| `Cabin`       | Cabin number                         |
| `Embarked`    | Port of embarkation                  |

---

## 🛠️ Technologies & Libraries Used

The project was implemented using **Python** and the following libraries:

* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Data preprocessing and standardization

---

## 🔄 Data Preprocessing Steps

### 1. Importing and Exploring the Dataset

The Titanic dataset was first imported using Pandas.

The basic structure of the dataset was examined using functions such as:

* `head()`
* `info()`
* `describe()`
* `isnull().sum()`
* `dtypes`

This helped identify:

* Number of rows and columns
* Data types
* Missing values
* Numerical and categorical features
* Basic statistical information

---

### 2. Handling Missing Values

Missing values were identified using:

```python
df.isnull().sum()
```

The missing values were handled using appropriate imputation techniques.

For numerical columns such as `Age`, the **median** was used because it is less sensitive to extreme values.

For categorical columns such as `Embarked`, the **mode** was used.

The `Cabin` column contains a large number of missing values, so it can be dropped when it is not useful for the intended analysis.

Example:

```python
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

---

### 3. Encoding Categorical Features

Machine Learning algorithms generally require numerical input.

Therefore, categorical features such as `Sex` and `Embarked` were converted into numerical representations.

For example:

* `Sex`: Male → 0, Female → 1
* `Embarked`: converted using one-hot encoding

Example:

```python
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
```

This makes the categorical information suitable for Machine Learning algorithms.

---

### 4. Feature Scaling

Numerical features can have very different ranges. For example, `Age` and `Fare` have different scales.

To make numerical features comparable, standardization was applied using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[['Age', 'Fare']] = scaler.fit_transform(
    df[['Age', 'Fare']]
)
```

Standardization transforms the features so that they have approximately:

* Mean = 0
* Standard deviation = 1

---

### 5. Detecting and Handling Outliers

Outliers were visualized using **boxplots**.

Example:

```python
sns.boxplot(x=df['Fare'])
plt.show()
```

Boxplots were used to identify unusually high or low values in numerical features such as:

* `Age`
* `Fare`
* `SibSp`
* `Parch`

After identifying outliers, appropriate treatment was performed where necessary.

One common method is the **Interquartile Range (IQR)** method:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values outside these boundaries can be considered potential outliers.

---

## 📊 Visualization

Boxplots were used to visually inspect the distribution of numerical variables and identify potential outliers.

Example:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df[['Age', 'Fare']])
plt.xticks(rotation=45)
plt.show()
```

Visualization makes it easier to understand the distribution of the data before and after preprocessing.

---

## 📁 Project Structure

```text
Titanic-Data-Preprocessing/
│
├── dataset/
│   └── Titanic-Dataset.csv
│
├── notebooks/
│   └── Titanic_Preprocessing.ipynb
│
├── images/
│   └── boxplots.png
│
├── README.md
│
└── requirements.txt
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-link>
```

### 2. Navigate to the project directory

```bash
cd Titanic-Data-Preprocessing
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or, if a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

### 4. Open the Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/Titanic_Preprocessing.ipynb
```

---

## ✅ Key Learning Outcomes

Through this task, I learned and practiced:

* Loading and exploring datasets using Pandas
* Understanding dataset structure and data types
* Identifying missing values
* Handling missing values using imputation
* Encoding categorical variables
* Scaling and standardizing numerical features
* Detecting outliers using boxplots
* Understanding the IQR method for outlier detection
* Preparing raw data for Machine Learning models
* Using Python libraries for data preprocessing and visualization

---

## Conclusion

The Titanic dataset was successfully preprocessed by performing data exploration, missing-value treatment, categorical encoding, feature scaling, and outlier detection.

This task provided a practical understanding of the **data preprocessing pipeline**, which is an essential foundation for building reliable Machine Learning models.

---

## 👨‍💻 Internship

**AI/ML Internship – ELEBATE LABS**

**Day 1 | Task 1 – Titanic Dataset Preprocessing**
## Author: Sarvesh Kumar
---
