# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px  # for visualization
from PIL import Image
# load the model from disk
import pickle
from streamlit_option_menu import option_menu
import seaborn as sns
import firebase_admin
from firebase_admin import credentials, firestore, auth

if not firebase_admin._apps:
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate(
        "C:/Users/DELL/OneDrive/Desktop/EmployeeChurnPrediction-main/employee--churn-firebase-adminsdk-m6oac-cc62eb5e3a.json")
    firebase_admin.initialize_app(cred)

# Initialize Firestore client globally
db = firestore.client()

# Single Prediction App
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)


def app():
    def navigation():
        try:
            path = st.query_params()['p'][0]
        except Exception as e:
            st.error('Please use the main app.')
            return None
    return navigation


def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


# Defining bar chart function
def bar(feature, df):
    # Groupby the categorical feature
    temp_df = df.groupby([feature, 'Predicted_target']).size().reset_index()
    temp_df = temp_df.rename(columns={0: 'Count'})
    # Calculate the value counts of each distribution and its corresponding Percentages
    value_counts_df = df[feature].value_counts().to_frame().reset_index()
    categories = [cat[1][0] for cat in value_counts_df.iterrows()]
    # Calculate the value counts of each distribution and its corresponding Percentages
    num_list = [num[1][1] for num in value_counts_df.iterrows()]
    div_list = [element / sum(num_list) for element in num_list]
    percentage = [round(element * 100, 1) for element in div_list]
    # Defining string formatting for graph annotation
    # Numeric section
    def num_format(list_instance):
        formatted_str = ''
        for index, num in enumerate(list_instance):
            if index < len(list_instance) - 2:
                formatted_str = formatted_str + f'{num}%, '  # append to empty string(formatted_str)
            elif index == len(list_instance) - 2:
                formatted_str = formatted_str + f'{num}% & '
            else:
                formatted_str = formatted_str + f'{num}%'
        return formatted_str

    # Categorical section
    def str_format(list_instance):
        formatted_str = ''
        for index, cat in enumerate(list_instance):
            if index < len(list_instance) - 2:
                formatted_str = formatted_str + f'{cat}, '
            elif index == len(list_instance) - 2:
                formatted_str = formatted_str + f'{cat} & '
            else:
                formatted_str = formatted_str + f'{cat}'
        return formatted_str

    # Running the formatting functions
    num_str = num_format(percentage)
    cat_str = str_format(categories)

    # Setting graph framework
    fig = px.bar(temp_df, x=feature, y='Count', color='Predicted_target',
                 title=f'Attrition rate by {feature}', barmode="group", color_discrete_sequence=["green", "red"])

    fig.add_annotation(
        text=f'Value count of distribution of {cat_str} are<br>{num_str} percentage respectively.',
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=0,
        y=1.14,
        bordercolor='black',
        borderwidth=1)
    fig.update_layout(
        # margin space for the annotations on the right
        margin=dict(r=100),
    )

    fig


# Function to register new user
def register(company_name, hr_name, email, password, confirm_password):
    if password != confirm_password:
        st.error("Password and confirm password do not match.")
        return False

    try:
        user = auth.create_user(
            email=email,
            password=password,
            display_name=hr_name,
            email_verified=False  # Assuming email is not verified by default
        )

        if user:
            st.success("Registration successful!")
            # Add additional user information to Firestore including company name
            db.collection('users').document(email).set({
                'company_name': company_name,
                'hr_name': hr_name,
                'email': email
            })
            return True

    except ValueError as e:
        st.error(f"Registration failed: {e}")
        return False

    except firebase_admin.auth.EmailAlreadyExistsError:
        st.error("User already exists.")
        return False

    # If registration is successful, set logged_in to False to redirect to login page
    return False


# Function to authenticate user
def authenticate(email, password):
    try:
        user = auth.get_user_by_email(email)
        if user:
            st.success("Login successful!")
            # Redirect to prediction app page
            st.session_state.logged_in = True
            st.session_state.email = email
            return True
    except auth.UserNotFoundError:
        st.error("User not found.")
    except firebase_admin.auth.EmailAlreadyExistsError:
        st.error("The user with the provided email already exists. Please use a different email.")
    except auth.InvalidPasswordError:
        st.error("Invalid password.")
    return False

    
# Function to store prediction data in Firestore
def store_prediction_data(input_data, prediction_result):
    try:
        # Convert non-string values to strings
        input_data_str = {str(key): str(value) for key, value in input_data.items()}
        prediction_result_str = "Employee Leave" if prediction_result == [1] else "Employee Stay"
        email = st.session_state.email

        if email:
            # Add data to Firestore collection under user's email
            db.collection('users').document(email).collection('predictions').add({
                'input_data': input_data_str,
                'prediction_result': prediction_result_str
            })
            st.success("Prediction data stored successfully")

            st.success("Prediction data stored successfully in Firestore under user's email!")
        else:
            st.error("User's email is not found in session state.")
    except Exception as e:
        st.error(f"Failed to store prediction data in Firestore: {e}")

def main():  # Set custom CSS styles

    session_state = st.session_state
    if 'logged_in' not in session_state:
        session_state['logged_in'] = False

    if not session_state['logged_in']:
        page = st.sidebar.radio("Login or Register", ["Login", "Register"], key="login_register")
        if page == "Login":
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if authenticate(email, password):
                    session_state['logged_in'] = True
                    session_state['email'] = email
        elif page == "Register":  
            company_name = st.text_input("Company Name", key="register_company_name")
            hr_name = st.text_input("HR Name", key="register_hr_name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
            if st.button("Register"):
                if register(company_name, hr_name, email, password, confirm_password):
                    # Automatically login after successful registration
                    session_state['email'] = email
                    

    elif session_state['logged_in']:
        if st.title('Employee Churn Prediction App'):
            sign_out = st.sidebar.button("Sign Out")

# Handle sign-out functionality
            if sign_out:
    # Reset session state variables and logged_in status
                session_state['logged_in'] = False
                session_state['email'] = None

    # Redirect to login/register page
                st.session_state.logged_in = False
            selected = option_menu("", ['Explore', 'Single', 'Batch', 'About'],
            icons=['house', 'cloud-upload', "list-task", "info-circle"], 
            menu_icon="cast", default_index=0, orientation="horizontal",
            styles={
                    "container": {"padding": "20px!important", "background-color": "#fafafa"},  # Increase padding
                    "icon": {"color": "orange", "font-size": "22px"},  # Increase icon size
                    "nav-link": {"font-size": "22px", "text-align": "left", "margin":"10px", "--hover-color": "#eee"},  # Increase font size
                    "nav-link-selected": {"background-color": "green"},
                    })

        if selected == "Explore":

            st.sidebar.write("## Explore The Dataset")
            st.sidebar.write("### Select an option from below: ")
        
            #df = pd.read_csv(selected_dataset)
            df = pd.read_csv(r"HR_Dataset.csv")
            selected_dataset = df
        #  dataset_type = check_dataset_category(selected_dataset)
        # st.info(f'Dataset Type: {dataset_type}')

        # Show the dimension of the dataframe
            if st.sidebar.checkbox("Show number of rows and columns"):
                st.subheader('Number of rows and columns')
                st.warning(f'Rows: {df.shape[0]}')
                st.info(f'Columns: {df.shape[1]}')
        
        # Distribution of Attrition
            if st.sidebar.checkbox('Distribution of Attrition'):
                st.subheader('Distribution of Attrition')
                target_instance = df["Predicted_target"].value_counts().to_frame()
                target_instance = target_instance.reset_index()
                target_instance = target_instance.rename(columns={'index': 'Category'})
                fig = px.pie(target_instance, values='Predicted_target', names='Category', color_discrete_sequence=["green", "red"])
                fig

        # display the dataset
            if st.sidebar.checkbox("Show Dataset"):
                st.write("#### The Dataset")
                rows = st.number_input("Enter the number of rows to view", min_value=0,value=5)
                if rows > 0:
                    st.dataframe(df.head(rows))  

        # Show dataset description
            if st.sidebar.checkbox("Show description of dataset"):
                st.write("#### The Description the Dataset")
                st.write(df.describe())
                
        # Select columns to display
            if st.sidebar.checkbox("Show dataset with selected columns"):
            # get the list of columns
                columns = df.columns.tolist()
                st.write("#### Explore the dataset with column selected:")
                selected_cols = st.multiselect("Select desired columns", columns)
                if len(selected_cols) > 0:
                    selected_df = df[selected_cols]
                    st.dataframe(selected_df)            

        # check the data set columns
            if st.sidebar.checkbox("Show dataset columns"): 
                st.write("#### Columns of the dataset:")
                st.write(df.columns)

        # counts how many of each class we have
            if st.sidebar.checkbox("Show dataset types"): 
                col_vals = df.select_dtypes(include=["object"])
                num_vals = df.select_dtypes(exclude=["object"])
                st.write("## Types of the dataset")
            
                st.warning("##### Numerical Values:")
                st.write(num_vals.columns)
                st.write("")
                st.warning("##### Categorical Values:")
                st.write(col_vals.columns)
            
   
            if st.sidebar.checkbox("Overlook the dataset"):
                st.write('### Overlook the dataset') 
                columns = df.columns.tolist()
                selected_cols = st.selectbox("Select the attribute:", columns)
                plt.title(selected_cols)
                fig = plt.figure(figsize=(10,6))
                plt.xticks(rotation=60, fontsize=10)   
                sns.countplot(x=selected_cols, data=df)
                st.pyplot(fig)  
            
            
            if st.sidebar.checkbox('Compare with Predicted Target'):
                st.write('#### Compare the dataset Against the Predicted Target')
                df_drop = df.drop('Predicted_target', axis=1)  # Remove the predicted target from the comparison
                columns = df_drop.columns.tolist()
                selected_cols = st.selectbox("Select the attribute", columns)
                plt.title(selected_cols)             
                bar(selected_cols, df)

            

        elif selected == "Single":             
            # User input fields
            e1 = st.slider("Employee satisfaction level", 0.0, 1.0, 0.5)
            e2 = st.slider("Last evaluation score", 0.0, 1.0, 0.5)
            e3 = st.slider("Number of projects assigned to", 1, 10, 5)
            e4 = st.slider("Average monthly hours worked", 50, 300, 150)
            e5 = st.slider("Time spent at the company", 1, 10, 3)
            e6 = st.radio("Work accident", ['No', 'Yes'], key="work_accident")
            e7 = st.radio("Promotion in the last 5 years", ['No', 'Yes'], key="promotion_last_5_years")
            e8 = st.selectbox("Department name", ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'], key="department_name")
            e9 = st.selectbox("Salary category", ['low', 'medium', 'high'], key="salary_category")

            # Convert radio button values to binary
            e6 = 1 if e6 == 'Yes' else 0
            e7 = 1 if e7 == 'Yes' else 0

            # Predict button
            if st.button("Predict"):
                sample = pd.DataFrame({
                    'satisfaction_level': [e1],
                    'last_evaluation': [e2],
                    'number_project': [e3],
                    'average_montly_hours': [e4],
                    'time_spend_company': [e5],
                    'Work_accident': [e6],
                    'promotion_last_5years': [e7],
                    'departments': [e8],
                    'salary': [e9]
                })
                data = {
                    'Satisfaction Level': [e1],
                    'Last Evaluation': [e2],
                    'Projects Assigned': [e3],
                    'Hours Worked': [e4],
                    'Company Time': [e5],
                    'Accident': [e6],
                    'Promotion': [e7],
                    'Department': [e8],
                    'Salary Category': [e9]
                }
                  # Convert dictionary to DataFrame for display
                features_df = pd.DataFrame.from_dict([data])
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.write('Overview of input is shown below')
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.dataframe(features_df)
                prediction_result = pipeline.predict(sample)
                if prediction_result == 1:
                    st.write("An employee may leave the organization.")
                else:
                    st.write("An employee may stay with the organization.")
  
                store_prediction_data(sample.to_dict(), prediction_result.tolist())
                

        elif selected == "Batch":
    
            # Content for App 2
            # Function to process data
            def process_data(data):
                # Load the model and perform predictions
                with open('pipeline.pkl', 'rb') as f:
                    pipeline = pickle.load(f)
                
                result = pipeline.predict(data)
                
                # Assign predictions based on result
                y_pred = ["An employee may leave the organization." if pred == 1 
                            else "An employee may stay with the organization." for pred in result]
                
                # Add predicted target to the data
                data['Predicted_target'] = y_pred
                
                 # Visualize predictions
                fig, ax = plt.subplots()
                data['Predicted_target'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_aspect('equal')
                ax.set_title('Predicted Employee Churn')
                st.pyplot(fig)
                return data

            # Streamlit app
            st.title("Predicting Employee Churn Using Machine Learning")

            # Button to upload CSV file
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

            if uploaded_file is not None:
                try:
                    # Load data from CSV
                    data = pd.read_csv(uploaded_file)
                    data.columns = data.columns.str.replace('\n', '')
                    data.rename(columns={'Departments ': 'departments'}, inplace=True)
                    data = data.drop_duplicates()
                    
                    # Process the data
                    processed_data = process_data(data)
                    
                    # Save the processed data to a CSV file
                    st.write("Processed Data:")
                    st.write(processed_data)
                    st.write("Saving the processed data...")
                    processed_data.to_csv('processed_data.csv', index=False)
                    st.success("Data saved successfully!")
                except Exception as e:
                    st.error(f"Failed to open file: {e}")


        elif selected == "About":
            st.title("About This App")
            st.write("Welcome to the Employee Churn Prediction App!")
            st.write("This Streamlit app is designed to help HR professionals and organizations predict employee churn.")
            st.write("Employee churn, also known as employee turnover, is a critical issue for businesses as it can lead to increased recruitment costs, loss of productivity, and negative impacts on company culture.")
            st.write("By leveraging machine learning algorithms, this app aims to provide insights into which employees are at risk of leaving, allowing HR departments to take proactive measures to retain valuable talent.")
            st.write("Here are some key features of the app:")
            st.write("- **Dataset Exploration:** Visualize the HR dataset, explore its distributions, and gain insights into employee attrition trends.")
            st.write("- **Single Prediction:** Input individual employee data to get predictions about whether they are likely to leave the organization.")
            st.write("- **Batch Prediction:** Upload a CSV file containing employee data to make predictions in bulk, enabling HR departments to analyze large datasets efficiently.")
            st.write("- **Firebase Integration:** User registration, authentication, and prediction data storage are managed using Firebase, ensuring secure and personalized user experiences.")
            st.write("The app is built using Streamlit for the user interface, pandas and scikit-learn for data processing and machine learning, Plotly and Matplotlib for data visualization, and Firebase for user authentication and data storage.")
            st.write("Feel free to explore the app's functionalities and make use of its features to gain insights into employee churn within your organization!")
    
            st.write("### Employee Churn Visualization")
            st.write("Below is an example of a visualization that can be generated using the app, showing the predicted employee churn distribution:")
    
            # Sample image graph
            image = Image.open("employee_churn.png")
            st.image(image, caption="Predicted Employee Churn Distribution", use_column_width=True)





if __name__ == "__main__":
     main()
