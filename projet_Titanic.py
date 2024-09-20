import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load data with caching
@st.cache_data
def load_data(nrows):
    data = pd.read_csv('train.csv', nrows=nrows)
    data.columns = data.columns.str.lower()
    return data


# User inputs
# name = st.text_input("Enter your name")
# age* = st.number_input("Enter your age", min_value=0, max_value=100)

# if name and age:
#     st.write(f"Hello, {{name}}!")

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
    
    #####
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
import plotly.express as px
if st.checkbox('age distribution charts'):
    if st.checkbox ('area chart'):
        st.caption('Age distribution area chart')
        fig = px.area(data['age'].value_counts().reset_index(), x='age', y='count')
        st.plotly_chart(fig)

    if st.checkbox('bar chart'):
        st.caption('Age distribution bar chart')
        fig = px.bar(data['age'].value_counts().reset_index(), x='age', y='count')
        st.plotly_chart(fig)

    if st.checkbox ('scatter plot'):
        st.caption('Scatter plot age')
        fig = px.scatter(data['age'].value_counts().reset_index(), x='age', y='count')
        st.plotly_chart(fig)

if st.checkbox( 'fare distribution'):
    st.caption('Fare distribution')
    fig = px.bar(data['fare'].value_counts().reset_index(), x='fare', y='count')
    st.plotly_chart(fig)

# line chart and scatter
if st.checkbox('sibsp, parch, and pclass distribution' ):
    st.caption('Line sibsp, parch, pclass')
    fig = px.line(data['sibsp'].value_counts().reset_index(), x='sibsp', y='count')
    st.plotly_chart(fig)
    fig = px.line(data['parch'].value_counts().reset_index(), x='parch', y='count')
    st.plotly_chart(fig)
    fig = px.line(data['pclass'].value_counts().reset_index(), x='pclass', y='count')
    st.plotly_chart(fig) 
    
  ####
X = data.drop(columns=['survived','passengerid','pclass','sibsp','ticket','cabin','embarked','name','parch'])
y = data["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

st.write(X_train)  
    ###
    # Data cleaning
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



# Create a histogram figure
selected_columns = st.multiselect("Variable presentation after featuring", ["age", "fare", "sex_female","sex_male"])
for column in selected_columns:
    fig = px.histogram(data, x=column, title=f"Variable presentation after featuring ({{column}})")
    st.subheader(f'{column} : Variable presentation after featuring')
    st.plotly_chart(fig)

for column in selected_columns:
    fig2 = px.histogram(data, x=column, y='survived', title=f"relation survived avec ({{column}})")
    st.subheader(f'{column} : relation avec survived')
    st.plotly_chart(fig2)

# Data splitting
from sklearn.model_selection import train_test_split
### Transforming data
fit_transform(Simp,Ohe,X_train)
Transform(Simp,Ohe,X_test)

###Decision Tree
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as mt

# Initialize and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#  predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#  accuracy
accuracy_train = mt.accuracy_score(y_train, y_pred_train)
accuracy_test = mt.accuracy_score(y_test, y_pred_test)

#  results
st.header('Decision Tree Classifier Results')
st.text(f'The first 10 predictions: {y_pred_train[:10]}')
st.text(f'Training accuracy: {accuracy_train}')
st.text(f'Test accuracy: {accuracy_test}')
### dynamic decision Tree

#  user inputs
age = st.slider('Age', min_value=0.0, max_value=100.0, step=2.0)
fare = st.slider('Fare', min_value=0.0, max_value=600.0, step=10.0)
sex = st.selectbox('Your gender is:', options=['male', 'female'])
df = pd.DataFrame({'age': [age], 'fare': [fare], 'sex': [sex]})
st.write('Input DataFrame:', df)
features = Transform(Simp, Ohe, df)
st.write('Transformed Features:', features)
#age_fair_transformed=Transform(Simp,Ohe,df)

#pd.DataFrame( OneHotEncoder.fit_transform(df[['sex']]).toarray(), columns= OneHotEncoder.get_feature_names(['sex']))

# pd.DataFrame( OneHotEncoder.fit_transform(df[['sex']]).toarray(), columns= OneHotEncoder.get_feature_names(['sex']))
#features = pd.concat([transformed_df,], axis=1)
#features=Transform(Simp,Ohe,df)

# df[['age', 'fare']] = si.transform(df[['age', 'fare']])  
# df['sex']=oh.transform(df[['sex']],)
# features = df.concat['age','fare','sex']
st.write(features)
#  prediction 
# if st.button('Predict'):
#     prediction = model.predict(features)
#     st.write(f'Prediction: {prediction}')
### SVC

from sklearn.svm import SVC
sv = SVC(kernel="poly",degree=2,C=1)
sv.fit(X_train,y_train)
y_pred_train = sv.predict(X_train)
cr=pd.DataFrame(mt.classification_report(y_pred_train,y_train,output_dict=True))
st.write(cr)
y_pred_test=sv.predict(X_test)

st.write('Model: Arbre de décision')
st.write('SV: méthode SVC')

# Create a selectbox to choose the model
model_choice = st.selectbox('Choose a model', ['Model', 'SV'])
# Create a single button
if st.button('Predict'):
    if model_choice == 'Model':
        prediction = model.predict(features)
        st.write(f'Prediction with Model: {prediction}')
    elif model_choice == 'SV':
        prediction = sv.predict(features)
        st.write(f'Prediction with SV: {prediction}')