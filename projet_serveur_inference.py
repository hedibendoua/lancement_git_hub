


####
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
#import seaborn as sns

##******##
name = st.text_input("Enter your name")
age = st.number_input("Enter your age", min_value=0, max_value=100)

if name and age:
    st.write(f"Hello, {name}! ")


    st.title('Fisrt App GMC')
    st.subheader('welcom to my first application')
    st.subheader('TITANIC Dataset')

    DATE_COLUMN = 'date/time'
    data = pd.read_csv('train.csv')
    def load_data(nrows):
        data = pd.read_csv('train.csv', nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data  

    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data = load_data(10000)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    #st.download_button("Download file", data)
    ###
    # Initialize a session state variable
    if 'data_shown' not in st.session_state:
        st.session_state.data_shown = False
    # Create a button
    if st.button("Show me Data"):
        # Toggle the data shown flag
        st.session_state.data_shown = not st.session_state.data_shown
    # Show or hide the data based on the flag
    if st.session_state.data_shown:
        st.write(data)
    else:
        st.write("")


    ###
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)
    ###
    if st.checkbox('Show raw head data'):
        st.subheader('Raw data')
        st.write(data.head())
    if st.checkbox('Show raw age describe'):
        st.subheader('age')
        st.write(data['age'].describe())
    if st.checkbox('Show raw columns describe'):
        st.subheader('columns')
        st.write(data.describe())

    #Draw graph

    if st.checkbox('age distribution charts'):
        if st.checkbox ('area chart'):
            st.subheader('age distribution area chart')
            st.area_chart(data['age'].value_counts().reset_index(),x='age',y='count')
        if st.checkbox('bar chart'):
            st.subheader('age distribution bar chart')
            st.bar_chart(data['age'].value_counts().reset_index(),x='age',y='count')
        if st.checkbox ('scatter plot'):
            st.subheader ('scatter plot age')
            fig = px.scatter(data['age'].value_counts().reset_index(),x='age', y='count')
            st.plotly_chart(fig)
    if st.checkbox( 'fare distribution'):
        st.subheader('fare distribution')
        st.bar_chart(data['fare'].value_counts().reset_index(),x='fare',y='count')
    # line chart and scatter
    if st.checkbox('sibsp,parch, and pclass distribution' ):
        st.subheader (' line sibsp, parch, pclass')
        st.line_chart(data['sibsp'].value_counts().reset_index(),x='sibsp',y='count')
        st.line_chart(data['parch'].value_counts().reset_index(),x='parch',y='count')
        st.line_chart(data['pclass'].value_counts().reset_index(),x='pclass',y='count')
        #  event = st.line_chart_chart(data['sibsp'].value_counts().reset_index(),x='sibsp',y='count',on_select="rerun")

    ###

    data.dropna(subset=['age','fare','sex'],inplace=True)

    from sklearn.model_selection import train_test_split

    X = data[['age','fare','sex']]
    y = data["survived"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

    ## Encoding sex
    from sklearn.preprocessing import OneHotEncoder
    oh = OneHotEncoder(sparse_output=False)
    oh.fit(X_train[["sex"]])
    X_train.loc[:,oh.get_feature_names_out()] = oh.transform(X_train.loc[:,["sex"]])
    X_test.loc[:,oh.get_feature_names_out()] = oh.transform(X_test.loc[:,["sex"]])
    X_train.drop(columns=["sex"],inplace=True)
    X_test.drop(columns=["sex"],inplace=True)
    st.write(X_train.head())


    ###

    from sklearn.impute import SimpleImputer
    si = SimpleImputer()
    si.fit(X_train[["age"]])
    X_train.loc[:,["age"]] = si.transform(X_train.loc[:,["age"]])
    X_test.loc[:,["age"]] = si.transform(X_test.loc[:,["age"]])


    # Create a histogram figure
    selected_columns = st.multiselect("Variable presentation after featuring", ["age", "fare", "sex"])
    for column in selected_columns:
        fig = px.histogram(data, x=column, title=f"Variable presentation after featuring ({column})")
        st.subheader(f'{column} : Variable presentation after featuring')
        st.plotly_chart(fig)
    for column in selected_columns:
        fig2 = px.histogram(data, x=column,y='survived', title=f"relation survived avec ({column})")
        st.subheader(f'{column} : relation avec survived')
        st.plotly_chart(fig2)
    # fig = px.histogram(data, x="age", title="Imputation Avec Moyenne Simple")
    # # Display the histogram
    # st.subheader('age : Imputation Avec Moyenne Simple')
    # st.plotly_chart(fig)
    # ### Decesion Tree
    from sklearn.tree import DecisionTreeClassifier
    model= DecisionTreeClassifier()
    model.fit(X_train,y_train)
    st.write(model.fit(X_train,y_train))
    y_pred_train= pd.DataFrame(model.predict(X_train))
    st.write(f' les dix premiers résultats',y_pred_train.head(10),output_dict=True)
    import sklearn.metrics as mt
    mt.accuracy_score(y_train, y_pred_train)
    st.write(f'accuracy train : {mt.accuracy_score(y_train, y_pred_train)}')
    y_pred_test= model.predict(X_test)
    mt.accuracy_score(y_test, y_pred_test)
    st.write(f'accuracy test : {mt.accuracy_score(y_test, y_pred_test)}')
    ####
    from sklearn.svm import SVC
    sv = SVC(kernel="poly",degree=2,C=1)
    sv.fit(X_train,y_train)
    y_pred_train = sv.predict(X_train)
    cr=pd.DataFrame(mt.classification_report(y_pred_train,y_train,output_dict=True))
    st.write(cr)

    ###
    import streamlit as st
    # Initialize an empty list to store chat messages
    chat_messages = []
    # Create a text input field for the user to type their message
    user_message = st.text_input("Type your message:")
    # Create a button to send the message
    if st.button("Send"):
        # Add the user's message to the chat messages list
        chat_messages.append(user_message)
        # Clear the text input field
        st.session_state.user_message = ""
    # Display the chat messages
    for message in chat_messages:
        st.write(message)
    # Optional: Add a clear chat button
    if st.button("Clear Chat"):
        chat_messages.clear()
        st.write("Chat cleared!")
        
    ######  RANDOM FOREST CLASSIFIER ####
    import streamlit as st
    from sklearn.ensemble import RandomForestClassifier
  
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)


# Create inputs for user
    user_input_age = st.number_input('age', min_value=0, max_value=99)
    user_input_sex_male = st.selectbox('sex_male', options=[0, 1])
    user_input_sex_female = st.selectbox('sex_female', options=[0, 1])
    user_input_fare = st.slider('fare', min_value=0, max_value=600)

# Button to run prediction
    if st.button('Predict'):
        # Create a DataFrame from the user's input
        input_df = pd.DataFrame({
            'feature1': [user_input_age],
            'feature2': [user_input_sex_male],
            'feature3':[user_input_sex_female],
            'feature4':[user_input_fare]})
        
        
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        clf.predict(X_test)

# Now you can use the classifier to make predictions
        
        prediction = model.predict(input_df)
        
        # Display the prediction
        st.write(f'Prediction: {prediction}')
    
    
    
    
    
    ####    
    else:
        st.write("Please enter your name and age.")