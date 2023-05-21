import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
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



# Display the boxplot using Streamlit
st.title('Boxplot')
fig_box, ax_box = plt.subplots()
df.boxplot(ax=ax_box)
st.pyplot(fig_box)

# Display the histogram using Streamlit
st.title('Histogram')
fig_hist, ax_hist = plt.subplots(figsize=(15, 8))
df.hist(ax=ax_hist)
st.pyplot(fig_hist)

# Save the histogram plot as a file
plt.savefig('histogram.png')

# Create a pie chart of the number of patients with hypertension
fig_pie_hyp, ax_pie_hyp = plt.subplots()
hypertension_counts = df["hypertension"].value_counts()
ax_pie_hyp.pie(hypertension_counts, labels=hypertension_counts.index)
ax_pie_hyp.set_title("Percentage of Patients with Hypertension")
st.pyplot(fig_pie_hyp)

# Create a bar chart of the number of patients by work type
fig_bar_work, ax_bar_work = plt.subplots()
work_type_counts = df["work_type"].value_counts()
ax_bar_work.bar(work_type_counts.index, work_type_counts.values)
ax_bar_work.set_xlabel("Work Type")
ax_bar_work.set_ylabel("Number of Patients")
ax_bar_work.set_title("Number of Patients by Work Type")
st.pyplot(fig_bar_work)

# Create a bar chart of the number of patients by residence type
fig_bar_res, ax_bar_res = plt.subplots()
residence_type_counts = df["Residence_type"].value_counts()
ax_bar_res.bar(residence_type_counts.index, residence_type_counts.values)
ax_bar_res.set_xlabel("Residence Type")
ax_bar_res.set_ylabel("Number of Patients")
ax_bar_res.set_title("Number of Patients by Residence Type")
st.pyplot(fig_bar_res)

# Create a scatter plot of the average glucose level and the BMI of the patients
fig_scatter, ax_scatter = plt.subplots()
ax_scatter.scatter(df["avg_glucose_level"], df["bmi"])
ax_scatter.set_xlabel("Average Glucose Level")
ax_scatter.set_ylabel("BMI")
ax_scatter.set_title("Scatter Plot of Average Glucose Level and BMI")
st.pyplot(fig_scatter)

# Create a pie chart of the number of patients with hypertension
fig_pie_hyp2, ax_pie_hyp2 = plt.subplots()
hypertension_counts = df["hypertension"].value_counts()
ax_pie_hyp2.pie(hypertension_counts, labels=hypertension_counts.index)
ax_pie_hyp2.set_title("Percentage of Patients with Hypertension")
st.pyplot(fig_pie_hyp2)

# Create a bar chart of the number of patients by gender
fig_bar_gender, ax_bar_gender = plt.subplots()
gender_counts = df["gender"].value_counts()
ax_bar_gender.bar(gender_counts.index, gender_counts.values)
ax_bar_gender.set_xlabel("Gender")
ax_bar_gender.set_ylabel("Number of Patients")
ax_bar_gender.set_title("Number of Patients by Gender")
st.pyplot(fig_bar_gender)


st.markdown(""" 
Frist insight
a bar chart that shows the percentage of patients who are male and female and who have had a stroke. The bar chart will look like this:
[Image of a bar chart showing the percentage of patients who are male and female and who have had a stroke. The bar chart shows that 48.6% of patients are male and 51.4% of patients are female. The bar chart also shows that 79.8% of patients did not have a stroke and 20.2% of patients had a stroke.]
The bar chart shows that the majority of patients in the dataset are female and that the majority of patients did not have a stroke. This information can be used to gain a better understanding of the relationships between the attributes and the target variable.
""")




st.title('After Model preprocessing and cross-validation and ')

st.title("Accuracy is Equal = 94.5 %")


# Load the model
model = pickle.load(open('modelf.pkl', 'rb'))

# Create a form to input the features
st.title('Stroke Prediction')

age = st.slider('Age', 18, 100)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
avg_glucose_level = st.slider('Average Glucose Level', 70, 300)
bmi = st.slider('BMI', 18, 60)
work_type = st.selectbox('Work Type:', ['Govt Job', 'Never Worked', 'Private', 'Self-employed', 'Children'])
residence_type = st.selectbox('Residence Type:', ['Rural', 'Urban'])
smoking_status = st.selectbox('Smoking Status:', ['Unknown', 'Formerly Smoked', 'Never Smoked', 'Smokes'])
ever_married = st.selectbox('Ever Married:', ['No', 'Yes'])
gender = st.selectbox('Gender:', ['Male', 'Female'])

# Create a function to make predictions and display the output
encoder = LabelEncoder()

def predict_stroke():
    categorical_features = [hypertension, heart_disease, work_type,
                            residence_type, smoking_status,
                            ever_married,
                            gender]

    # Fit the encoder on the categorical features
    encoder.fit(categorical_features)

    # Transform the categorical features
    encoded_features = encoder.transform(categorical_features)

    # Reshape numerical features to match the dimensions of encoded features
    numerical_features = np.array([age, avg_glucose_level, bmi]).reshape(1, -1)

    # Combine all features
    features = np.concatenate((numerical_features, encoded_features.reshape(1, -1)), axis=1)

    # Make sure the number of features matches the model's expectations
    if features.shape[1] != model.coef_.shape[1]:
        st.error(f"The number of features ({features.shape[1]}) does not match the model's expected number of features ({model.coef_.shape[1]}).")
        return

    # Predict the stroke
    prediction = model.predict(features)

    # Display the prediction
    if prediction[0] == 0:
        st.write('Have Stroke!')
    else:
        st.write('Not have Stroke, Congrats!')

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

st.title("Powered by Eng:Youssef Azam")
st.write("Mobile phone: 01222627208")
st.write("LinkedIn: https://www.linkedin.com/in/youssef-azam-a36816231")


