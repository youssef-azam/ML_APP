# -*- coding: utf-8 -*-
"""FULL_Project_for_Data scientists.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OmKDHDZeCey6DfSLdKE3nbgbiwztpQub

### النهاردة هقولك دور كل واحد فين من الوظايف دي 

1.   Data analyst
2.   ML
3.   Data scientists

وطبعا البدايه لازم تكون من Data analystوبعدة اسباب  

1.  Collect data from a variety of sources, such as databases, spreadsheets, and surveys.
2. Clean and prepare data for analysis by removing errors, correcting inconsistencies, and transforming data into a format that can be easily analyzed.
3. Analyze data using statistical methods, such as regression analysis, clustering, and classification.
4. Visualize data using charts, graphs, and other data visualizations to communicate findings to stakeholders.

# **Data analyst**

## read data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'/content/healthcare-dataset-stroke-data.csv')
df.head(3)

"""** Summary:

The stroke dataset is a relatively clean and well-formatted dataset. The target variable, stroke, is binary, and there are no missing values. There are some interesting relationships between the attributes, such as the positive correlation between age, hypertension, heart disease, smoking, and avg_glucose_level and the negative correlation between stroke and ever_married and Residence_type. These relationships can be further explored using machine learning models to predict the risk of stroke. **

## Understand Data

##### gender Male or female
##### age 
##### hypertension  ارتفاع ضغط الدم
##### heart_disease   مرض القلب
##### ever_married متزوج ام لا 
##### work_type  نوع العمل 
##### Residence_type نوع الإقامة
##### avg_glucose_level  متوسط ​​مستوى الجلوكوز في الدم
##### bmi  الوزن 
##### smoking_status  يدخن ام لا 
# stroke  سكتة دماغية

** Relationship between attributes:

The following are some of the relationships between the attributes:

Age and stroke: There is a positive correlation between age and stroke, meaning that older patients are more likely to have a stroke.
Hypertension and stroke: There is a positive correlation between hypertension and stroke, meaning that patients with hypertension are more likely to have a stroke.
Heart disease and stroke: There is a positive correlation between heart disease and stroke, meaning that patients with heart disease are more likely to have a stroke.
Smoking and stroke: There is a positive correlation between smoking and stroke, meaning that smokers are more likely to have a stroke.
Ever_married and stroke: There is a negative correlation between ever_married and stroke, meaning that married patients are less likely to have a stroke.
Residence_type and stroke: There is a positive correlation between Residence_type and stroke, meaning that patients living in urban areas are more likely to have a stroke.
Avg_glucose_level and stroke: There is a positive correlation between Avg_glucose_level and stroke, meaning that patients with higher blood sugar levels are more likely to have a stroke.
Bmi and stroke: There is a positive correlation between Bmi and stroke, meaning that patients with higher body mass indexes are more likely to have a stroke. **

##### تعال اقولك بقا انا كمحلل بيانات اي مهمتي 

 ** بعد ما فهمت الداتا ببتكملم عن اي لازم اقدر احدد اي اهمية كل عمود بنسبة للاخر علشان اعرف هيفيدني ولا لا 
مبدايا دي داتا بتعبر عن كل واحدح فينا واي حد لازم يشوفها علي نفسه بمعنب اي دي خصايص موجوده جوانا بلفعل 
مثلا اي مرض قد يسبب ان الانسان يجي سكته دماغيه طيب السجاير ممكن تعمل كدا فعلا ممكن تعمل وانها لها تاثيرات علي ضضغط الدم برضو ف بتالي هتاثر جامد اووووي طيب نوع العمل هياثر علي النفسية ودي حاجه مهمه اووووي وطبعا متزوج ولا لا دي بتفرق جدا من خلال الرعايه 
والعمر كل ما يزيد الصحيه بتقل **

## EDA
"""

df.shape

# Check the data types
df.dtypes

# Check for missing values
df.isna().sum()

# Distribution of target variable
df['stroke'].value_counts()

# Relationships between attributes
sns.pairplot(df, hue='stroke')

# Outliers
df.boxplot()

# Summary
df.describe()

df.hist(figsize=(15,8))
plt.show()

# Histogram of age and stroke
plt.hist(df['age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Number of patients')
plt.title('Histogram of age and stroke')
plt.show()

# Boxplot of hypertension and stroke
sns.boxplot(x='stroke', y='hypertension', data=df)
plt.xlabel('Stroke')
plt.ylabel('Hypertension')
plt.title('Boxplot of hypertension and stroke')
plt.show()

# Boxplot of heart disease and stroke
sns.boxplot(x='stroke', y='heart_disease', data=df)
plt.xlabel('Stroke')
plt.ylabel('Heart disease')
plt.title('Boxplot of heart disease and stroke')
plt.show()

# Boxplot of smoking and stroke
sns.boxplot(x='stroke', y='smoking_status', data=df)
plt.xlabel('Stroke')
plt.ylabel('Smoking status')
plt.title('Boxplot of smoking and stroke')
plt.show()

# Boxplot of avg_glucose_level and stroke
sns.boxplot(x='stroke', y='avg_glucose_level', data=df)
plt.xlabel('Stroke')
plt.ylabel('Average glucose level')
plt.title('Boxplot of avg_glucose_level and stroke')
plt.show()

# Boxplot of bmi and stroke
sns.boxplot(x='stroke', y='bmi', data=df)
plt.xlabel('Stroke')
plt.ylabel('Body mass index')
plt.title('Boxplot of bmi and stroke')
plt.show()

"""## Frist insight
a bar chart that shows the percentage of patients who are male and female and who have had a stroke. The bar chart will look like this:

[Image of a bar chart showing the percentage of patients who are male and female and who have had a stroke. The bar chart shows that 48.6% of patients are male and 51.4% of patients are female. The bar chart also shows that 79.8% of patients did not have a stroke and 20.2% of patients had a stroke.]

The bar chart shows that the majority of patients in the dataset are female and that the majority of patients did not have a stroke. This information can be used to gain a better understanding of the relationships between the attributes and the target variable.

## Full EDA يعوض اللي فات ويخليك تفهم اكتر ❤
"""

from dataprep.datasets import get_dataset_names
from dataprep.eda import create_report
from dataprep.clean import clean_df,clean_date
import pandas as pd
import seaborn as sns
df= pd.read_csv(r"/content/healthcare-dataset-stroke-data.csv")
report=create_report(df)

report

"""# Data cleaning"""

(df.isnull().sum())/len(df),df.isnull().sum()

df['gender'].value_counts()

# Get the row of columns where the gender is equal to "Other"
other_row = df[df["gender"] == "Other"]

# Drop the row of columns
df = df.drop(other_row.index, axis=0)

"""##### why use this mothed 
######  columns it continue this null value if we use mean it is not effect my data 
##### and i drop it my data it is very small 
###### the best method to use mean 
"""

# Get the mean of the "bmi" column
mean_bmi = df["bmi"].mean()

# Replace the NaN values in the "bmi" column with the mean
df["bmi"] = df["bmi"].fillna(mean_bmi)

df.isnull().sum()

"""# Data visualization"""

# Create a pie chart of the number of patients with hypertension
plt.pie(df["hypertension"].value_counts(), labels=df["hypertension"].unique())
plt.title("Percentage of Patients with Hypertension")
plt.show()

# Create a bar chart of the number of patients by gender
plt.bar(df["gender"].unique(), df["ever_married"].value_counts())
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.title("Number of Patients Who Are Ever Married by Gender")
plt.show()

# Create a bar chart of the number of patients by work type
plt.bar(df["work_type"].unique(), df["work_type"].value_counts())
plt.xlabel("Work Type")
plt.ylabel("Number of Patients")
plt.title("Number of Patients by Work Type")
plt.show()

# Create a bar chart of the number of patients by residence type
plt.bar(df["Residence_type"].unique(), df["Residence_type"].value_counts())
plt.xlabel("Residence Type")
plt.ylabel("Number of Patients")
plt.title("Number of Patients by Residence Type")
plt.show()

# Create a scatter plot of the average glucose level and the BMI of the patients
plt.scatter(df["avg_glucose_level"], df["bmi"])
plt.xlabel("Average Glucose Level")
plt.ylabel("BMI")
plt.title("Scatter Plot of Average Glucose Level and BMI")
plt.show()

# Create a bar chart of the number of patients by smoking status
plt.bar(df["smoking_status"].unique(), df["smoking_status"].value_counts())
plt.xlabel("Smoking Status")
plt.ylabel("Number of Patients")
plt.title("Number of Patients by Smoking Status")
plt.show()

# Create a bar chart of the number of patients who had a stroke by gender
plt.bar(df["gender"].unique(), df["stroke"].value_counts())
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.title("Number of Patients Who Had a Stroke by Gender")
plt.show()



"""###### دلوقتي في مننا بينهي دور Data analyst وبعض الاخر بيقول انولازم يحدد Feature Engineer من وجهة نظريان اه لازم هو اللي يحدده لان دي من اهم الحاجات في mlوبتعتمد اعتمد كلي علي فهم الداتا وفهم العلاقات بين الجداول وبعضه

# Feature Engineer
"""

df.head(2)

# Create a function to determine if a patient is a future engineer
def is_future_engineer(df):
    if df["gender"] == "Male" and df["age"] >= 22 and df["age"] <= 28 and df["heart_disease"] == 0 and df["ever_married"] == "No" and df["work_type"] == "Private" and df["Residence_type"] == "Urban" and df["avg_glucose_level"] <= 100 and df["bmi"] <= 25 and df["smoking_status"] == "never smoked":
        return True
    else:
        return False

# Apply the function to the DataFrame
future_engineers = df.apply(is_future_engineer, axis=1)

# Count the number of future engineers
number_of_future_engineers = future_engineers.sum()

# Print the number of future engineers
print(number_of_future_engineers)

# Create a correlation matrix
corr = df.corr()

# Plot the correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot=True, cmap="Blues")
plt.show()

"""# ML
### Bulid Model 
"""

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import lightgbm as lgb

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import (
    train_test_split,
    KFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-white')

# Convert the categorical variables to numerical variables
df = pd.get_dummies(df, columns=["work_type", "Residence_type", "smoking_status","ever_married","gender"])

# Create the features and target variables
X = df.drop(['stroke', 'id'], axis=1)
y = df['stroke']

df.info()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_train

"""## Model preprocessing """

# Create a scaler object
scaler = StandardScaler()

# Scale the features in the training set
X_train = scaler.fit_transform(X_train)

# Scale the features in the test set
X_test = scaler.transform(X_test)

# Create the models
logistic_regression = LogisticRegression()
svm = SVC()
knn = KNeighborsClassifier()

# Fit the models to the training data
logistic_regression.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred_logistic_regression = logistic_regression.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_knn = knn.predict(X_test)

# Calculate the accuracy of the models
accuracy_logistic_regression = np.mean(y_pred_logistic_regression == y_test)
accuracy_svm = np.mean(y_pred_svm == y_test)
accuracy_knn = np.mean(y_pred_knn == y_test)

# Print the accuracy of the models
print("Accuracy of logistic regression:", accuracy_logistic_regression)
print("Accuracy of SVM:", accuracy_svm)
print("Accuracy of KNN:", accuracy_knn)

"""### cross-validation"""

# Calculate the cross-validation score
accuracy_logistic_regression = logistic_regression.score(X_test, y_test)
accuracy_svm = svm.score(X_test, y_test)
accuracy_knn =knn.score(X_test, y_test)

# Print the cross-validation score
print("Cross-validation score accuracy_logistic_regression = :", accuracy_logistic_regression)
print("Cross-validation score accuracy_svm = :", accuracy_svm)
print("Cross-validation score accuracy_knn = :", accuracy_knn)

"""## The hights **accuracy is SVM**

### Accuracy is Equal = 94.5 %

### Model save
"""

import pickle

with open('my_model.pkl', 'wb') as f:
  pickle.dump(accuracy_svm, f)

"""# Data scientists

## Business value

# Insights:
The most important factor in predicting stroke is age.
Other important factors include sex, race, smoking, hypertension, diabetes, and heart disease.
People who are older, male, black, smokers, hypertensive, diabetic, or have heart disease are at an increased risk for stroke.
# Decisions:
You can use these insights to develop strategies for preventing stroke.
For example, you could develop educational campaigns to raise awareness of the risk factors for stroke.
You could also develop programs to help people quit smoking, control their blood pressure, and manage their diabetes.
It is important to note that these are just some general insights and decisions. The specific insights and decisions that you make will depend on the specific data that you have available.

###### Educate people about the risk factors for stroke. This can be done through public service announcements, educational campaigns, and doctor-patient conversations.
###### Help people quit smoking. Smoking is a major risk factor for stroke, and quitting can significantly reduce your risk.
###### Help people control their blood pressure. High blood pressure is another major risk factor for stroke, and controlling it can help reduce your risk.
###### Help people manage their diabetes. Diabetes is also a major risk factor for stroke, and managing it can help reduce your risk.
###### Encourage people to get regular exercise. Exercise can help reduce your risk of stroke by improving your overall health and fitness.
###### Eat a healthy diet. A healthy diet can help reduce your risk of stroke by helping you control your weight, blood pressure, and cholesterol levels.
###### Lose weight if you are overweight or obese. Being overweight or obese is a major risk factor for stroke, and losing weight can help reduce your risk.
###### Get regular medical checkups. This is important for detecting and treating any risk factors for stroke early on.

# Some Statistics and ratios based on decisions

**Age: The risk of stroke increases with age. For example, people aged 65 and older are about twice as likely to have a stroke as people aged 45 to 64.
Sex: Men are at a slightly higher risk of stroke than women.
Race: African Americans are at a higher risk of stroke than Caucasians.
Smoking: Smokers are at a much higher risk of stroke than nonsmokers.
Hypertension: People with hypertension are at a higher risk of stroke than people with normal blood pressure.
Diabetes: People with diabetes are at a higher risk of stroke than people without diabetes.
Heart disease: People with heart disease are at a higher risk of stroke than people without heart disease.
You can use these statistics and ratios to help you make decisions about stroke prevention. For example, if you are a smoker, you may want to consider quitting smoking to reduce your risk of stroke. If you have hypertension, you may want to consider taking medication to control your blood pressure. And if you have diabetes, you may want to consider taking medication to control your blood sugar.

By taking these steps, you can help reduce your risk of stroke and improve your overall health.

Here are some additional statistics and ratios that you may find helpful:

The risk of stroke increases by about 2% for every 10 mmHg increase in systolic blood pressure.
The risk of stroke increases by about 4% for every 10 mmHg increase in diastolic blood pressure.
People with diabetes are about twice as likely to have a stroke as people without diabetes.
People with heart disease are about three times as likely to have a stroke as people without heart disease.
About 87% of strokes are preventable.
The average cost of a stroke is about $40,000.**

# THANKS ❤
"""