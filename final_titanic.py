import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt
from sklearn.svm import SVC
import numpy as np
@st.cache_data
def load_data(nrows):
    data = pd.read_csv('train.csv', nrows=nrows)
    data.columns = data.columns.str.lower()
    return data
st.title('First App GMC')
st.subheader('Welcome to my first application')
st.subheader('TITANIC Dataset')
data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text('Loading data...done!')
X = data.drop(columns=['survived', 'passengerid', 'pclass', 'sibsp', 'ticket', 'cabin', 'embarked', 'name', 'parch'])
y = data["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
st.write(X_train)
# commentaire test GIT
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
# Initialize the OneHotEncoder
Ohe = OneHotEncoder(sparse_output=False)
Simp = SimpleImputer()
##fit and transform data
# Fit the imputer and transform the 'age', 'fare', and 'sex' columns in the data
def fit_transform(Simp, Ohe, data):
    Simp.fit(data[['age', 'fare']])
    data.loc[:, ['age', 'fare']] = Simp.transform(data[['age', 'fare']])
    Ohe.fit(data[['sex']])
    data.loc[:, Ohe.get_feature_names_out()] = Ohe.transform(data[['sex']])
    data.drop(columns=["sex"], inplace=True)
def Transform(Simp, Ohe, data):
    data.loc[:, ['age', 'fare']] = Simp.transform(data[['age', 'fare']])
    data.loc[:, Ohe.get_feature_names_out()] = Ohe.transform(data[['sex']])
    data.drop(columns=["sex"], inplace=True)
fit_transform(Simp,Ohe,X_train)
Transform(Simp,Ohe,X_test)    
### Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
# accuracy
accuracy_train = mt.accuracy_score(y_train, y_pred_train)
accuracy_test = mt.accuracy_score(y_test, y_pred_test)
st.write("Train accuracy:", accuracy_train)
st.write("Test accuracy:", accuracy_test)
#SVC
sv = SVC(kernel="poly",degree=2,C=1)
sv.fit(X_train,y_train)
y_pred_train = sv.predict(X_train)
cr=pd.DataFrame(mt.classification_report(y_pred_train,y_train,output_dict=True))
st.write(cr)
y_pred_test=sv.predict(X_test)
st.write('Model: Arbre de décision')
st.write('SV: méthode SVC')
###inference
age = st.slider('Age', min_value=0.0, max_value=100.0, step=2.0)
fare = st.slider('Fare', min_value=0.0, max_value=600.0, step=10.0)
sex = st.selectbox('Your gender is:', options=['male', 'female'])
df = pd.DataFrame({'age': [age], 'fare': [fare], 'sex': [sex]})
st.write('Input DataFrame:', df)
Transform(Simp, Ohe, df)
st.write('Transformed Features:', df)
# Create a selectbox to choose the model
model_choice = st.selectbox('Choose a model', ['Model', 'SV'])
# Create a single button
if st.button('Predict'):
    if model_choice == 'Model':
        prediction = model.predict(df)
        st.write(f'Prediction with Model: {prediction}')
    elif model_choice == 'SV':
        prediction = sv.predict(df)
        st.write(f'Prediction with SV: {prediction}')