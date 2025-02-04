import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#Configuring our streamlit webpage
st.set_page_config(
    page_title="Heart Failure Prediction",
    layout="centered"
)

def load_model():
    try:
        model = pickle.load(open('Heart_failure_model.pkl','rb'))
        return model
    except Exception as e:
        st.error(f"Error Loading the model:{e}")
        return None

def predict_failure(model,features):
    try:
        prediction = model.predict([features])[0]
        return prediction
    except Exception as e:
        st.error(f"Error Making the prediction:{e}")
        return None

def main():
    st.title("Heart Failure Chance Predictor")
    st.write("Enter below details to predict Heart failure chances")

    with st.form("prediction_form"):
        age = st.slider("AGE",0,120,100)
        creatinine_phosphokinase = st.slider("Creatinine Phosphokinase",0,1200,1000)
        ejection_fraction=st.slider("Ejection Fraction",1,100,70)
        anaemia = st.radio("Anaemia",["Yes","No"])
        diabetes = st.radio("Diabetes",["Yes","No"])
        sex = st.radio("Sex",["Male","Female"])
        smoking = st.radio("Smoking",["Yes","No"])
        high_blood_pressure = st.radio("High Blood Pressure",["Yes","No"])
        serum_creatinine = st.slider("Serum Creatinine",0,10,8)
        serum_sodium = st.slider("Serum Sodium",40,160,100)
        platelets = st.slider("Platelets",10000,1200000,1000000)
        
        submitted = st.form_submit_button("Predict Heart Failure Chance")
        
        if submitted:
            model = load_model()

            if model:
                anaemia = 1 if anaemia == "Yes" else 0
                diabetes = 1 if diabetes == "Yes" else 0
                high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0
                sex = 1 if sex == "Female" else 0
                smoking = 1 if smoking == "Yes" else 0
                features = [age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking]
                
                prediction = predict_failure(model,features)
                
                if prediction is not None:
                    if prediction >= 1:
                        prediction = 1
                    elif prediction <= 0:
                        prediction = 0
                    st.success(f"Your prediction Chance of Heart Failure is {prediction*100:.2f}%")
                    
                    if prediction >= 0.8:
                        st.write("Strong chances of Heart Failure")
                    elif prediction >=0.6:
                        st.write("Medium chances of Heart Failure")
                    else:
                        st.write("Low Chances of Heart Failure")

                    st.info("Disclaimer: This prediction is a tool to provide a general risk assessment and should not be considered a diagnosis. It is based on statistical analysis of population data and may not accurately reflect your individual risk. It is essential to consult with a qualified healthcare professional for a proper diagnosis and personalized treatment plan. Do not make any changes to your current treatment or medication regimen based solely on this prediction.")

    st.markdown("Factors affecting  :")
    st.write(""" Age:  The risk of heart failure increases with age.  As we get older, the heart muscle can weaken, and other age-related health issues can contribute.   

Anaemia (Low Red Blood Cell Count): Anaemia can put a strain on the heart.  The heart has to work harder to pump oxygen-carrying blood throughout the body, which can weaken it over time.   

Creatinine Phosphokinase (CPK):  Elevated CPK levels can indicate muscle damage, including damage to the heart muscle.  High CPK can be a sign of a recent heart attack or other heart-related problems.   

Diabetes: Diabetes significantly increases the risk of heart failure.  High blood sugar can damage blood vessels, including those supplying the heart.  It can also lead to other conditions that increase heart failure risk, such as high blood pressure and obesity.   

Ejection Fraction: Ejection fraction measures the percentage of blood the left ventricle pumps out with each contraction.  A low ejection fraction is a key indicator of heart failure.  It means the heart isn't pumping blood effectively.   

High Blood Pressure (Hypertension):  High blood pressure puts a constant strain on the heart, making it work harder.  Over time, this can weaken the heart muscle and lead to heart failure.   

Platelets: Platelets are involved in blood clotting.  While necessary, an abnormally high platelet count can increase the risk of blood clots, which can lead to heart attacks and strokes, both of which can cause or worsen heart failure.   

Serum Creatinine: Serum creatinine is a measure of kidney function.  Kidney problems can both be a cause and a consequence of heart failure.  Impaired kidney function can worsen heart failure, and heart failure can reduce blood flow to the kidneys, damaging them.   

Serum Sodium:  Sodium levels in the blood need to be carefully balanced.  Abnormal sodium levels can disrupt fluid balance in the body, which can put a strain on the heart.  In heart failure, the body may retain excess fluid, leading to low sodium levels (hyponatremia), which can be dangerous.   

Sex:  While both men and women can develop heart failure, some risk factors and symptoms can differ slightly between the sexes.  For example, women are more likely to develop heart failure with preserved ejection fraction.   

Smoking: Smoking is a major risk factor for heart disease in general, including heart failure.  It damages blood vessels, increases blood pressure, and reduces the oxygen-carrying capacity of the blood.   


             """)
    
if __name__ == "__main__":
    main()
    
    