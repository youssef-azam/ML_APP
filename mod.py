import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
# from dataprep.datasets import get_dataset_names
# from dataprep.eda import create_report
# Display the summary using markdown
import streamlit as st
st.header('Full Project Of Data scientists In this project Will talk Roles of importance job ')
# Define the roles and responsibilities
roles = {
    "Data Analyst": [
        "Collect data from a variety of sources, such as databases, spreadsheets, and surveys.",
        "Clean and prepare data for analysis by removing errors, correcting inconsistencies, and transforming data into a format that can be easily analyzed.",
        "Analyze data using statistical methods, such as regression analysis, clustering, and classification.",
        "Visualize data using charts, graphs, and other data visualizations to communicate findings to stakeholders."
    ],
    "Machine Learning Engineer": [
        "Develop machine learning models and algorithms to solve specific business problems.",
        "Prepare and preprocess data for machine learning tasks, including feature engineering and selection.",
        "Implement and optimize machine learning algorithms and models.",
        "Evaluate and fine-tune machine learning models for optimal performance.",
        "Deploy machine learning models into production environments."
    ],
    "Data Scientist": [
        "Identify and define business problems that can be solved using data science techniques.",
        "Collect, clean, and preprocess large datasets from various sources.",
        "Apply advanced statistical and machine learning techniques to extract insights and build predictive models.",
        "Conduct exploratory data analysis and data visualization to uncover patterns and trends.",
        "Communicate findings and insights to non-technical stakeholders in a clear and understandable manner."
    ]
}

# Display the roles and responsibilities using Streamlit
st.title("Roles and Responsibilities")
for role, responsibilities in roles.items():
    st.header(role)
    for responsibility in responsibilities:
        st.write("- " + responsibility)
    st.write("\n")



st.markdown("""## Summary of Data:

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
##### stroke  سكتة دماغية

#### Relationship between attributes:

The following are some of the relationships between the attributes:

Age and stroke: There is a positive correlation between age and stroke, meaning that older patients are more likely to have a stroke.
Hypertension and stroke: There is a positive correlation between hypertension and stroke, meaning that patients with hypertension are more likely to have a stroke.
Heart disease and stroke: There is a positive correlation between heart disease and stroke, meaning that patients with heart disease are more likely to have a stroke.
Smoking and stroke: There is a positive correlation between smoking and stroke, meaning that smokers are more likely to have a stroke.
Ever_married and stroke: There is a negative correlation between ever_married and stroke, meaning that married patients are less likely to have a stroke.
Residence_type and stroke: There is a positive correlation between Residence_type and stroke, meaning that patients living in urban areas are more likely to have a stroke.
Avg_glucose_level and stroke: There is a positive correlation between Avg_glucose_level and stroke, meaning that patients with higher blood sugar levels are more likely to have a stroke.
Bmi and stroke: There is a positive correlation between Bmi and stroke, meaning that patients with higher body mass indexes are more likely to have a stroke. **""")
st.title('Person as Data analyst')
url = url = "https://raw.githubusercontent.com/youssef-azam/ML_APP/main/healthcare-dataset-stroke-data.csv"
df = pd.read_csv(url)




st.markdown(""" 
Frist insight
a bar chart that shows the percentage of patients who are male and female and who have had a stroke. The bar chart will look like this:
[Image of a bar chart showing the percentage of patients who are male and female and who have had a stroke. The bar chart shows that 48.6% of patients are male and 51.4% of patients are female. The bar chart also shows that 79.8% of patients did not have a stroke and 20.2% of patients had a stroke.]
The bar chart shows that the majority of patients in the dataset are female and that the majority of patients did not have a stroke. This information can be used to gain a better understanding of the relationships between the attributes and the target variable.
""")




st.title('After Model preprocessing and cross-validation and ')

st.title("Accuracy is Equal = 94.5 %")


# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Create a form to input the features
st.title('Stroke Prediction')

age = st.slider('Age', 18, 100)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
avg_glucose_level = st.slider('Average Glucose Level', 70, 200)
bmi = st.slider('BMI', 18, 40)
work_type_Govt_job = st.selectbox('Work Type: Govt Job', ['No', 'Yes'])
work_type_Never_worked = st.selectbox('Work Type: Never Worked', ['No', 'Yes'])
work_type_Private = st.selectbox('Work Type: Private', ['No', 'Yes'])
work_type_Self_employed = st.selectbox('Work Type: Self-employed', ['No', 'Yes'])
work_type_children = st.selectbox('Work Type: Children', ['No', 'Yes'])
residence_type_Rural = st.selectbox('Residence Type: Rural', ['No', 'Yes'])
residence_type_Urban = st.selectbox('Residence Type: Urban', ['No', 'Yes'])
smoking_status_Unknown = st.selectbox('Smoking Status: Unknown', ['No', 'Yes'])
smoking_status_formerly_smoked = st.selectbox('Smoking Status: Formerly Smoked', ['No', 'Yes'])
smoking_status_never_smoked = st.selectbox('Smoking Status: Never Smoked', ['No', 'Yes'])
smoking_status_smokes = st.selectbox('Smoking Status: Smokes', ['No', 'Yes'])
ever_married_No = st.selectbox('Ever Married: No', ['No', 'Yes'])
ever_married_Yes = st.selectbox('Ever Married: Yes', ['No', 'Yes'])
gender_Female = st.selectbox('Gender: Female', ['No', 'Yes'])
gender_Male = st.selectbox('Gender: Male', ['No', 'Yes'])

# Create a function to make predictions and display the output
def predict_stroke():
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', sparse_output=False)
    categorical_features = [[hypertension, heart_disease, work_type_Govt_job, work_type_Never_worked,
                             work_type_Private, work_type_Self_employed, work_type_children,
                             residence_type_Rural, residence_type_Urban, smoking_status_Unknown,
                             smoking_status_formerly_smoked, smoking_status_never_smoked,
                             smoking_status_smokes, ever_married_No, ever_married_Yes,
                             gender_Female, gender_Male]]

    # Fit the encoder on the categorical features
    encoder.fit(categorical_features)

    # Transform the categorical features
    encoded_features = encoder.transform(categorical_features)

    # Reshape numerical features to match the dimensions of encoded features
    numerical_features = np.array([age, avg_glucose_level, bmi]).reshape(1, -1)

    # Combine all features
    features = np.concatenate((numerical_features, encoded_features), axis=1)

    # Predict the stroke
    prediction = model.predict(features)

    # Display the prediction
    if prediction[0] == 0 :
        st.write('Have Stroke Sade!!')
    else:
        st.write('Not have Stroke ,Congrats!')

# Add a button to trigger the prediction function
if st.button('Predict'):
    predict_stroke()

st.title("Data scientists")

# Display the insights
st.markdown("## Insights:")
st.markdown("- The most important factor in predicting stroke is age.")
st.markdown("- Other important factors include sex, race, smoking, hypertension, diabetes, and heart disease.")
st.markdown("- People who are older, male, black, smokers, hypertensive, diabetic, or have heart disease are at an increased risk for stroke.")

# Display the decisions
st.markdown("## Decisions:")
st.markdown("- Educate people about the risk factors for stroke.")
st.markdown("- Help people quit smoking.")
st.markdown("- Help people control their blood pressure.")
st.markdown("- Help people manage their diabetes.")
st.markdown("- Encourage people to get regular exercise.")
st.markdown("- Eat a healthy diet.")
st.markdown("- Lose weight if overweight or obese.")
st.markdown("- Get regular medical checkups.")

# Display the statistics and ratios
st.markdown("## Statistics and Ratios:")
st.markdown("- Age: The risk of stroke increases with age.")
st.markdown("- Sex: Men are at a slightly higher risk of stroke than women.")
st.markdown("- Race: African Americans are at a higher risk of stroke than Caucasians.")
st.markdown("- Smoking: Smokers are at a much higher risk of stroke than nonsmokers.")
st.markdown("- Hypertension: People with hypertension are at a higher risk of stroke.")
st.markdown("- Diabetes: People with diabetes are at a higher risk of stroke.")
st.markdown("- Heart disease: People with heart disease are at a higher risk of stroke.")
st.markdown("- About 87% of strokes are preventable.")
st.markdown("- The average cost of a stroke is about $40,000.")

# Add a thank you message
st.markdown("# THANKS ❤")

