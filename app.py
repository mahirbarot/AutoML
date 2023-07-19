import streamlit as st
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import io
import os
import json,pickle
import base64
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import warnings
from sklearn.impute import SimpleImputer

# Ignore SettingWithCopyWarning from pandas
warnings.simplefilter(action="ignore", category=Warning)

with st.sidebar:
    st.title("AutoML")
    
    #inserted code is here
    st.title("Print Current Webpage")

    # Button to print the current webpage


    choice = st.radio("Navigation", ["Upload","Profiling","ML","Deployement"])
    


encodings = ['utf-8','utf-16','utf-32','latin-1','ascii','cp1252','cp437','cp850','cp852','cp855','cp856','cp857','cp858','cp860','cp861','cp862','cp863','cp864','cp865','cp866','cp869','cp874','cp875','cp932','cp949','cp950','gb2312','gbk','euc-jp','euc-kr'
]


if os.path.exists("sourcedata1.csv"):
    df = pd.read_csv("sourcedata1.csv",index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        # Initialize a flag to track successful read
        successful_read = False
        for encoding in encodings:
            try:
                df = pd.read_csv(file, index_col=None, encoding=encoding)
                successful_read = True
                break  # Exit the loop if successful read
            except UnicodeDecodeError:
                continue  # Try the next encoding if decoding error occurs

        if successful_read:
            df.to_csv("sourcedata1.csv", index=None)
            st.dataframe(df)
        else:
            st.error("Failed to read the CSV file. Please check the file encoding.")

if choice == "Profiling":
    st.title("Auto EDA")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

column_names = [   "No","NO","no", "RowNumber",    "CustomerId",    "Index",    "ID",    "SerialNumber",    "RecordId",    "Timestamp",    "CreatedAt",    "UpdatedAt",    "UniqueIdentifier",    "Counter",    "SequenceNumber",    "Description",    "Notes",    "Comments",    "Remarks",    "Metadata",    "Source",    "SourceId",    "SourceSystem",    "SourceFile",    "SourceDate",    "SourceType",    "SourceName",    "SourceURL",    "SourceCode",    "SourceLocation",    "SourceCategory",    "SourceStatus",    "SourceVersion"]

# Define the label encoder object
label_encoder = LabelEncoder()

output = []
if choice == "ML":
    st.title("Machine Learning Models")

    # Choose between Regression and Classification
    model_type = st.selectbox("Select Model Type", ["Regression", "Classification"])

    target = st.selectbox("Select your target", df.columns)

    try:
        # Remove irrelevant columns based on column_names and "id" in feature names
        irrelevant_columns = [col for col in df.columns if col.lower() in column_names or "id" in col.lower()]
        df_filtered = df.drop(columns=irrelevant_columns)    
        
        # Drop the target column from df_filtered
        df_filtered = df_filtered.drop(columns=[target])
        
        # Convert categorical columns to numerical values using one-hot encoding
        categorical_cols = df_filtered.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df_filtered, columns=categorical_cols)
        
        # Define X (features) and y (target)
        X = df_encoded
        y = df[target]
        
        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Apply SelectKBest feature selection
        select_kbest = SelectKBest(k='all')
        X_kbest = select_kbest.fit_transform(X_imputed, y)
        
        # Get the indices of the selected features
        selected_features = select_kbest.get_support(indices=True)
        
        # Create a new dataframe with the selected features and the target variable
        df_selected = pd.concat([pd.DataFrame(X_kbest, columns=[X.columns[i] for i in selected_features]), y], axis=1)
        if model_type == "Classification":
            from pycaret.classification import *
            # Convert the target column to numerical values using label encoding
            label_encoder.fit(df_selected[target])  # Fit the label encoder on the target column
            df_selected[target] = label_encoder.transform(df_selected[target])  # Transform the target column

            # Convert the selected dataframe to a dictionary
            df_selected_dict = df_selected.to_dict()

            if st.button("Generate!"):
                setup(df_selected_dict, target=target)
                setup_df = pull()
                st.info("Trying out Hyperparameters")
                st.dataframe(setup_df)
                best_model = compare_models()
                compare_df = pull()
                st.info("This is the ML Model")
                st.dataframe(compare_df)
                st.info(best_model)

                # Display accuracy of the best model for classification
                best_model_accuracy = pull().iloc[0]['Accuracy']
                st.info("Accuracy of the Best Model: {:.2f}%".format(best_model_accuracy * 100))

                # Extract range of each feature
                feature_ranges = {}
                for feature in df_selected.columns[:-1]:
                    feature_min = df_selected[feature].min()
                    feature_max = df_selected[feature].max()
                    feature_ranges[feature] = {"min": int(feature_min), "max": int(feature_max) + 1}

                # Convert DataFrame to dictionary with int values
                df_selected_dict = df_selected.astype(int).to_dict()

                # Save metadata including feature/column names, data types, and ranges
                metadata = {
                    "Features": df_selected.columns[:-1].tolist(),
                    "Target": target,
                    "DataTypes": dict(df_selected.dtypes.apply(lambda x: x.name).to_dict()),
                    "FeatureRanges": feature_ranges
                }
                with open("metadata.json", "w") as metadata_file:
                    json.dump(metadata, metadata_file)

                # Get the distinct target values
                target_values = df[target].unique()

                # Create a string of target values without repetitions
                target_values_str = ",".join(target_values)

                # Save the target values to a text file
                target_values_file = "target_values.txt"
                with open(target_values_file, "w") as f:
                    f.write(target_values_str)

                # Save the best model using pickle
                model_file_path = "best_model.pkl"
                with open(model_file_path, "wb") as model_file:
                    pickle.dump(best_model, model_file)

                st.success("Model saved successfully!")

                # Provide the model file for download to the user
                st.markdown("### Download Model")
                download_button_str = f'<a href="data:file/pkl;base64,{base64.b64encode(open(model_file_path, "rb").read()).decode()}" download="best_model.pkl">Download Model</a>'
                st.markdown(download_button_str, unsafe_allow_html=True)

                # Provide the target values file for download to the user
                st.markdown("### Download Target Values")
                target_values_download_button_str = f'<a href="data:file/txt;base64,{base64.b64encode(open(target_values_file, "rb").read()).decode()}" download="target_values.txt">Download Target Values</a>'
                st.markdown(target_values_download_button_str, unsafe_allow_html=True)

                # Information about downloading target values
                st.info("Note: The Target Values file contains the distinct values of the target feature. Please make sure to download and refer to this file for the corresponding labels.")
        else :
            from pycaret.regression import *
            if st.button("Generate!"):
                setup(df_selected, target=target)
                setup_df = pull()
                st.info("Trying out Hyperparameters")
                st.dataframe(setup_df)
                best_model = compare_models()
                compare_df = pull()
                st.info("This is the ML Model")
                st.dataframe(compare_df)
                st.info(best_model)

                # Get evaluation metric of the best model for regression
                best_model_metrics = pull().iloc[0]
                metric_name = 'R2'  # Choose the desired metric (e.g., R2, MAE, MSE, RMSE)
                best_model_metric = best_model_metrics[metric_name]
                st.info("Metric ({}) of the Best Model: {:.2f}".format(metric_name, best_model_metric))
                # Extract range of each feature
                feature_ranges = {}
                for feature in df_selected.columns[:-1]:
                    feature_min = df_selected[feature].min()
                    feature_max = df_selected[feature].max()+1
                    feature_ranges[feature] = {"min": int(feature_min), "max": int(feature_max)}

                # Convert DataFrame to dictionary with int values
                df_selected_dict = df_selected.astype(int).to_dict()

                # Save metadata including feature/column names, data types, and ranges
                metadata = {
                    "Features": df_selected.columns[:-1].tolist(),
                    "Target": target,
                    "DataTypes": dict(df_selected.dtypes.apply(lambda x: x.name).to_dict()),
                    "FeatureRanges": feature_ranges
                }
                with open("metadata.json", "w") as metadata_file:
                    json.dump(metadata, metadata_file)

                # Save the best model using pickle
                model_file_path = "best_model.pkl"
                with open(model_file_path, "wb") as model_file:
                    pickle.dump(best_model, model_file)

                st.success("Model saved successfully!")

                # Provide the model file for download to the user
                st.markdown("### Download Model")
                download_button_str = f'<a href="data:file/pkl;base64,{base64.b64encode(open(model_file_path, "rb").read()).decode()}" download="best_model.pkl">Download Model</a>'
                st.markdown(download_button_str, unsafe_allow_html=True)
    
    except KeyError:
        pass

if choice =="Deployement":
    st.title("Deployment")

    ch = st.selectbox("Select the type of problem", ["classification", "regression"])
    if ch=="classification":

        # Define the target labels file upload
        labels_file = st.file_uploader("Upload Target Labels File", type=".txt")

        # Check if a target labels file is uploaded
        if labels_file is not None:
            # Read the content of the labels file
            labels_content = labels_file.getvalue().decode('utf-8')

            st.info("If you have not generated the model from our service and you want to deploy your own created model, please first create a text file containing target labels, where each label is comma-separated.")

            # Split the labels by comma to get individual target names
            target_names = labels_content.split(',')

            # Close the target labels file uploader
            labels_file.close()

            st.success("Target Labels uploaded successfully!")

            # Convert target names to DataFrame
            target_names_df = pd.DataFrame(target_names, columns=['Target Name'])

        # Take input for the pkl model file
        pkl_file = st.file_uploader("Upload PKL Model File", type=".pkl")
        # Check if a file is uploaded
        if pkl_file is not None:
            # Load the pkl model
            best_model = pickle.load(pkl_file)

            # Close the file uploader
            pkl_file.close()

            st.success("Model uploaded successfully!")
        

            # Open and read the metadata JSON file
            with open("metadata.json", "r") as metadata_file:
                metadata = json.load(metadata_file)

            features = metadata['Features']
            data_types = metadata['DataTypes']
            feature_ranges = metadata['FeatureRanges']


            # Store user inputs
            user_inputs = {}

            for feature in features:
                data_type = data_types[feature]
                feature_range = feature_ranges[feature]

                if data_type == 'int64':
                    user_inputs[feature] = st.number_input(
                        label=feature,
                        value=int(feature_range['min']),
                        min_value=int(feature_range['min']),
                        max_value=int(feature_range['max']),
                        step=1
                    )
                elif data_type == 'float64':
                    user_inputs[feature] = st.number_input(
                        label=feature,
                        value=float(feature_range['min']),
                        min_value=float(feature_range['min']),
                        max_value=float(feature_range['max']),
                        step=0.01
                    )
                elif data_type == 'bool':
                    user_inputs[feature] = st.checkbox(label=feature)
                elif data_type == 'object':
                    user_inputs[feature] = st.text_input(label=feature)
                    tansform_value = label_encoder.transform(user_inputs[feature])
                    user_inputs[feature] = tansform_value
                elif data_type == 'datetime64':
                    user_inputs[feature] = st.date_input(label=feature)
                elif data_type == 'category':
                    options = ['Option 1', 'Option 2', 'Option 3']  # Replace with your actual options
                    user_inputs[feature] = st.selectbox(label=feature, options=options)
                elif data_type == 'timedelta64':
                    user_inputs[feature] = st.slider(
                        label=feature,
                        min_value=float(feature_range['min']),
                        max_value=float(feature_range['max']),
                        value=(float(feature_range['min']), float(feature_range['max']))
                    )
                elif data_type == 'period':
                    options = ['Period 1', 'Period 2', 'Period 3']  # Replace with your actual options
                    user_inputs[feature] = st.selectbox(label=feature, options=options)
                elif data_type == 'interval':
                    user_inputs[feature] = st.slider(
                        label=feature,
                        min_value=float(feature_range['min']),
                        max_value=float(feature_range['max']),
                        value=(float(feature_range['min']), float(feature_range['max']))
                    )
                elif data_type == 'complex':
                    user_inputs[feature] = st.text_input(label=feature)

            
        # st.write(target_names_df.iloc[1])


        if st.button("Predict"):
            # Prepare input data
            input_data = pd.DataFrame([user_inputs])

                # Make predictions
            predictions = best_model.predict(input_data)
            
            prediction = target_names_df.iloc[predictions]
            st.write(prediction)
        
    elif ch=="regression" : 
        # Take input for the pkl model file
        pkl_file = st.file_uploader("Upload PKL Model File", type=".pkl")
        # Check if a file is uploaded
        if pkl_file is not None:
            # Load the pkl model
            best_model = pickle.load(pkl_file)

            # Close the file uploader
            pkl_file.close()

            st.success("Model uploaded successfully!")
        

            # Open and read the metadata JSON file
            with open("metadata.json", "r") as metadata_file:
                metadata = json.load(metadata_file)

            features = metadata['Features']
            data_types = metadata['DataTypes']
            feature_ranges = metadata['FeatureRanges']


            # Store user inputs
            user_inputs = {}

            for feature in features:
                data_type = data_types[feature]
                feature_range = feature_ranges[feature]

                if data_type == 'int64':
                    user_inputs[feature] = st.number_input(
                        label=feature,
                        value=int(feature_range['min']),
                        min_value=int(feature_range['min']),
                        max_value=int(feature_range['max']),
                        step=1
                    )
                elif data_type == 'float64':
                    user_inputs[feature] = st.number_input(
                        label=feature,
                        value=float(feature_range['min']),
                        min_value=float(feature_range['min']),
                        max_value=float(feature_range['max']),
                        step=0.01
                    )
                elif data_type == 'bool':
                    user_inputs[feature] = st.checkbox(label=feature)
                elif data_type == 'object':
                    user_inputs[feature] = st.text_input(label=feature)
                elif data_type == 'datetime64':
                    user_inputs[feature] = st.date_input(label=feature)
                elif data_type == 'category':
                    options = ['Option 1', 'Option 2', 'Option 3']  # Replace with your actual options
                    user_inputs[feature] = st.selectbox(label=feature, options=options)
                elif data_type == 'timedelta64':
                    user_inputs[feature] = st.slider(
                        label=feature,
                        min_value=float(feature_range['min']),
                        max_value=float(feature_range['max']),
                        value=(float(feature_range['min']), float(feature_range['max']))
                    )
                elif data_type == 'period':
                    options = ['Period 1', 'Period 2', 'Period 3']  # Replace with your actual options
                    user_inputs[feature] = st.selectbox(label=feature, options=options)
                elif data_type == 'interval':
                    user_inputs[feature] = st.slider(
                        label=feature,
                        min_value=float(feature_range['min']),
                        max_value=float(feature_range['max']),
                        value=(float(feature_range['min']), float(feature_range['max']))
                    )
                elif data_type == 'complex':
                    user_inputs[feature] = st.text_input(label=feature)

            
        # st.write(target_names_df.iloc[1])


        if st.button("Predict"):
            # Prepare input data
            input_data = pd.DataFrame([user_inputs])

                # Make predictions
            predictions = best_model.predict(input_data)
            
            st.write(predictions)
