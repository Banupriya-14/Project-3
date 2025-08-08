import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"E:\MDTM40\Project3_Employee\Employee-Attrition - Employee-Attrition.csv")

#Initialize LableEncoder
with open(r'E:\MDTM40\Project3_Employee\labelencoder.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Load the model from file
with open(r'E:\MDTM40\Project3_Employee\randomforest_model.pkl', 'rb') as filename:
    loaded_model = pickle.load(filename)

with open('E:/MDTM40/Project3_Employee/scaler.pkl', 'rb') as scalerfile:
    scaler_model = pickle.load(scalerfile)

st.sidebar.title('🌐 Employee Attrition Analysis and Prediction')
selection = st.sidebar.radio('Go to',['🏠 Home','🎯 Predict Employee Attrition','📌 Insights'])

if selection == '🏠 Home':
    st.title('  🪪 Employee Attrition Analysis and Prediction')
    st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
    st.write("""Employee turnover poses a significant challenge for organizations, resulting in increased costs, reduced productivity, and team disruptions. Understanding the factors driving attrition and predicting at-risk employees is critical for effective retention strategies. This project aims to analyze employee data, identify key drivers of attrition, and build predictive models to support proactive decision-making in workforce management.
 """)
    st.markdown("Explore the dataset 🗂️")
    if st.button('Click Here'):
        with st.status("Fetching data") as status:
            time.sleep(1)
            status.update(label="Completed!", state="complete", expanded=False)
        df=pd.read_csv('E:\MDTM40\Project3_Employee\Employee-Attrition - Employee-Attrition.csv')
        st.dataframe(df)

if selection == '🎯 Predict Employee Attrition':
    st.header(" 🔍 Predict Employee Attrition")
    col1, col2 = st.columns(2, gap='large')
    with col1:
        age=st.slider("Age", min_value= 18, max_value=60)
        btravel=st.selectbox("Business Travel",['Travel_Rarely','Travel_Frequently','Non-Travel'])
        depart=st.selectbox("Department",['Sales','Research & Development', 'Human Resources'])
        distanceFromHome=st.slider("Distance FromHome", min_value= 1, max_value=100)
        education=st.selectbox("Education",[1,2,3,4,5])
        educationField=st.selectbox("Education Field",['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
        environmentSatisfaction=st.selectbox("Environment Satisfaction", [1,2,3,4,5])
        jobInvolvement=st.selectbox("Job Involvement", [1,2,3,4,5])
        percentSalaryHike=st.slider("Percent Salary Hike", min_value= 5, max_value=30)
        performanceRating=st.slider("Performance Rating", min_value= 1, max_value=5)
        maritalstatus=st.selectbox("Marital status", ['Single', 'Married', 'Divorced'])
        jobSatisfaction=st.slider("Job Satisfaction", min_value= 1, max_value=5)



    with col2:
        gender=st.selectbox("Gender",['Male','Female'])
        JobRole=st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager',
       'Sales Representative', 'Research Director', 'Human Resources'])
        totalWorkingYears=st.slider("Total Working Years",min_value= 1, max_value=60)
        monthlyIncome=st.number_input("Monthly Income",min_value= 1000, max_value=500000)
        monthlyRate=st.number_input("Monthly Rate",min_value= 1000, max_value=500000)
        numCompaniesWorked=st.slider("Number of Companies Worked",min_value= 1, max_value=20)
        overTime=st.radio("Over Time",['Yes', 'No'])
        workLifeBalance=st.slider("Work Life Balance", min_value= 1, max_value=5)
        yearsAtCompany=st.slider("Years At Company",min_value= 1, max_value=50)
        yearsInCurrentRole=st.slider("Years In CurrentRole",min_value= 1, max_value=50)
        yearsSinceLastPromotion=st.slider("Years Since Last Promotion",min_value= 1, max_value=50)
        yearsWithCurrManager=st.slider("Years With CurrManager",min_value= 1, max_value=50)
    
    businessTravel = encoders['BusinessTravel'].transform([btravel])[0]
    department = encoders['Department'].transform([depart])[0]
    EducationField = encoders['EducationField'].transform([educationField])[0]
    Gender = encoders['Gender'].transform([gender])[0]
    JobRole = encoders['JobRole'].transform([JobRole])[0]
    MaritalStatus = encoders['MaritalStatus'].transform([maritalstatus])[0]
    OverTime = encoders['OverTime'].transform([overTime])[0]


    data = (age,businessTravel,department,distanceFromHome,education,EducationField,environmentSatisfaction,jobInvolvement,jobSatisfaction,percentSalaryHike,
            performanceRating,MaritalStatus,Gender,JobRole,totalWorkingYears,monthlyIncome,monthlyRate,numCompaniesWorked,OverTime,workLifeBalance,
            yearsAtCompany,yearsInCurrentRole,yearsSinceLastPromotion,yearsWithCurrManager)
    
    left, middle, right = st.columns(3)

    if data and middle.button("🔍 Predict Attrition", type="secondary"):

        input_data = np.array(data).reshape(1,-1)


        prediction = loaded_model.predict(scaler_model.transform(input_data))

        if prediction[0]==0:
            st.balloons()
            st.success("💼 Great news! The employee is engaged and likely to remain with the organization.")
        else:
            st.warning("🚨Risk Alert: This employee may leave. Consider taking retention actions.")

if selection == '📌 Insights':

    st.header('📈 Insights')

    left, right = st.columns(2)    
    with left:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Department", hue="Attrition", ax=ax)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.countplot(data=df, x="MaritalStatus", hue="Attrition", ax=ax)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.countplot(data=df, y="EducationField", hue="Attrition", ax=ax)
        ax.set_title("Attrition Count by EducationField")
        st.pyplot(fig)


    with right:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Gender", hue="Attrition", ax=ax)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.countplot(data=df, x="BusinessTravel", hue="Attrition", ax=ax)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.countplot(data=df, y="JobRole", hue="Attrition", ax=ax)
        ax.set_title("Attrition Count by JobRole")
        st.pyplot(fig)





