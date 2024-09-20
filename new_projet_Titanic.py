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
# Function to load data with caching
@st.cache_data
def load_data(nrows):
    data = pd.read_csv('train.csv', nrows=nrows)
    data.columns = data.columns.str.lower()
    return data
# User inputs
name = st.text_input("Enter your name")
age = st.number_input("Enter your age", min_value=0, max_value=100)
if name and age:
    st.write(f"Hello, {name}!")
st.title('First App GMC')
st.subheader('Welcome to my first application')
st.subheader('TITANIC Dataset')
# Load data
data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text('Loading data...done!')
# Toggle data display
if st.checkbox("Show Data"):
    st.dataframe(data)
if st.checkbox('Show raw data'):
    st.caption('Raw data')
    st.dataframe(data)
if st.checkbox('Show raw head data'):
    st.caption('Raw data')
    st.dataframe(data.head())
if st.checkbox('Show raw age describe'):
    st.caption('Age')
    st.dataframe(data['age'].describe())
if st.checkbox('Show raw columns describe'):
    st.caption('Columns')
    st.dataframe(data.describe())
# Draw graph
if st.checkbox('age distribution charts'):
    if st.checkbox('area chart'):
        st.caption('Age distribution area chart')
        fig = px.area(data['age'].value_counts().reset_index(), x='age', y='count')
        st.plotly_chart(fig)
    if st.checkbox('bar chart'):
        st.caption('Age distribution bar chart')
        fig = px.bar(data['age'].value_counts().reset_index(), x='age', y='count')
        st.plotly_chart(fig)
    if st.checkbox('scatter plot'):
        st.caption('Scatter plot age')
        fig = px.scatter(data['age'].value_counts().reset_index(), x='age', y='count')
        st.plotly_chart(fig)
if st.checkbox('fare distribution'):
    st.caption('Fare distribution')
    fig = px.bar(data['fare'].value_counts().reset_index(), x='fare', y='count')
    st.plotly_chart(fig)
# line chart and scatter
if st.checkbox('sibsp, parch, and pclass distribution'):
    st.caption('Line sibsp, parch, pclass')
    fig = px.line(data['sibsp'].value_counts().reset_index(), x='sibsp', y='count')
    st.plotly_chart(fig)
    fig = px.line(data['parch'].value_counts().reset_index(), x='parch', y='count')
    st.plotly_chart(fig)
    fig = px.line(data['pclass'].value_counts().reset_index(), x='pclass', y='count')
    st.plotly_chart(fig)
# Data cleaning with fonction 


def fit_imputer(data, columns):
    si = SimpleImputer()
    si.fit(data[columns])
    return si

def transform_imputer(data, columns, imputer):
    data[columns] = imputer.transform(data[columns])
    return data

def fit_encoder(data, column):
    ohe = OneHotEncoder()
    ohe.fit(data[[column]].astype(str))
    return ohe

def transform_encoder(data, column, encoder):
    encoded = encoder.transform(data[[column]].astype(str))
    encoded_columns = encoder.get_feature_names_out([column])
    data[encoded_columns] = encoded.toarray()
    data.drop(columns=[column], inplace=True)
    return data

# si = SimpleImputer()
# data[['age', 'fare']] = si.fit_transform(data[['age', 'fare']])
# ohe=OneHotEncoder()
# sex_encoded = ohe.fit_transform(data[['sex']].astype(str),)
# sex_columns = ohe.get_feature_names_out(['sex'])
# data[sex_columns] = sex_encoded.toarray()
# data.drop(columns=["sex"], inplace=True)
# Create a histogram figure
selected_columns = st.multiselect("Variable presentation after featuring", ["age", "fare", "sex_female", "sex_male"])
for column in selected_columns:
    fig = px.histogram(data, x=column, title=f"Variable presentation after featuring ({column})")
    st.subheader(f'{column} : Variable presentation after featuring')
    st.plotly_chart(fig)
for column in selected_columns:
    fig2 = px.histogram(data, x=column, y='survived', title=f"relation survived avec ({column})")
    st.subheader(f'{column} : relation avec survived')
    st.plotly_chart(fig2)
# Data splitting
X = data.drop(columns=['survived', 'passengerid', 'pclass', 'sibsp', 'ticket', 'cabin', 'embarked', 'name', 'parch'])
y = data["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
st.write(X_train)
def fit_imputer(data, columns):
    si = SimpleImputer()
    si.fit(X_train[['age','fare']])
    return si
def transform_imputer(data, columns, imputer):
    X_test[['age','fare']] = imputer.transform(X_train['age','fare'])
    return data
def fit_encoder(data, column):
    ohe = OneHotEncoder()
    ohe.fit(X_train[['sex']].astype(str))
    return ohe

def transform_encoder(data, column, encoder):
    encoded = encoder.transform(X_test[['sex']].astype(str))
    encoded_columns = encoder.get_feature_names_out(['sex'])
    data[encoded_columns] = encoded.toarray()
    data.drop(columns=[column], inplace=True)
    return data

### Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
# accuracy
accuracy_train = mt.accuracy_score(y_train, y_pred_train)
accuracy_test = mt.accuracy_score(y_test, y_pred_test)
# results
st.header('Decision Tree Classifier Results')
st.text(f'The first 10 predictions: {y_pred_train[:10]}')
st.text(f'Training accuracy: {accuracy_train}')
st.text(f'Test accuracy: {accuracy_test}')
### Dynamic Decision Tree
st.title("Titanic Survival Predictor")
age = st.slider('Age', min_value=0.0, max_value=100.0, step=2.0)
fare = st.slider('Fare', min_value=0.0, max_value=600.0, step=10.0)
sex = st.selectbox('Your sex is:', options=['male', 'female'])
# Create a new dataframe with the input values
input_df = pd.DataFrame({'age': [age], 'fare': [fare], 'sex': [sex]})
# Transform the input data using the fitted instances
imputed_data = si.transform(input_df[["age", "fare"]])
input_df[["age", "fare"]] = imputed_data
sex_encoded = ohe.transform(input_df[["sex"]].astype(str))
sex_columns = ohe.get_feature_names_out(["sex"])
input_df_sex = pd.DataFrame(sex_encoded.toarray(), columns=sex_columns)
input_df.drop(columns=["sex"], inplace=True)
features = pd.concat([input_df, input_df_sex], axis=1)

st.write(features)

st.button('Predict')
prediction = model.predict(features)
st.write(f'Prediction: {prediction}')
### SVC
st.title("SVC Model training and test")
sv = SVC(kernel="poly", degree=2, C=1)
sv.fit(X_train, y_train)
y_pred_train = sv.predict(X_train)
cr = pd.DataFrame(mt.classification_report(y_pred_train, y_train, output_dict=True))
st.write('training prediction results',cr)
y_pred_test=sv.predict(X_test)
cr=pd.DataFrame(mt.classification_report(y_pred_test, y_test, output_dict=True))
st.write('test prediction results',cr)
###


# Create a Streamlit app
st.title("SVC Model Deployment")
st.write("Please enter your input data:")
# Calculate average age and fare for males and females
avg_age_male = round(X_train[X_train['sex_male'] == 1]['age'].mean())
avg_age_female = round(X_train[X_train['sex_female'] == 1]['age'].mean())
avg_fare_male = round(X_train[X_train['sex_male'] == 1]['fare'].mean())
avg_fare_female = round(X_train[X_train['sex_female'] == 1]['fare'].mean())
# Create input fields for the user
sex = st.selectbox("Sex:", ["male", "female"])
if sex == "male":
    age = st.number_input("Age:", min_value=0, max_value=100, value=avg_age_male)
    fare = st.number_input("Fare:", min_value=0, max_value=1000, value=avg_fare_male)
else:
    age = st.number_input("Age:", min_value=0, max_value=100, value=avg_age_female)
    fare = st.number_input("Fare:", min_value=0, max_value=1000, value=avg_fare_female)
# Create a button to make predictions
st.button("Make Prediction")
# Convert user input to a Pandas DataFrame
input_df = pd.DataFrame({"age": [age], "fare": [fare], "sex": [sex]})
# Featuring input data
  # assumed you have already defined the imputer
imputed_data = si.transform(input_df[["age", "fare"]])
input_df[["age", "fare"]] = imputed_data
  # assumed you have already defined the encoder = 
sex_encoded = ohe.transform(input_df[["sex"]].astype(str))
sex_columns = ohe.get_feature_names_out(["sex"])
input_df_sex = pd.DataFrame(sex_encoded.toarray(), columns=sex_columns)

input_df.drop(columns=["sex"], inplace=True)

features = pd.concat([input_df, input_df_sex], axis=1)
st.write(features)
st.write(input_df.columns())
# Make predictions with the trained model
y_pred = sv.predict(features)
    
    # Display the predicted class
st.write("Predicted class:", y_pred[0])
cr_test=pd.DataFrame(mt.classification_report(y_test,y_pred, output_dict=True))
st.write('test prediction results',cr_test)