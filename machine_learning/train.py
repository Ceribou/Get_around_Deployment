import os
import argparse
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Set tracking URI to your Heroku application
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

### MLFLOW Experiment setup
experiment_name="project_getaround"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

client = mlflow.tracking.MlflowClient()
run = client.create_run(experiment.experiment_id)

print("training model...")

# Call mlflow autolog
mlflow.sklearn.autolog(log_models=False) # We won't log models right away

# Load dataset
df = pd.read_csv('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv', index_col=0)
df[['private_parking_available', 'has_gps','has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']] = df[['private_parking_available', 'has_gps','has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']].apply(lambda x: "yes" if True else "no")

# Split dataset into X features and Target variable
target_variable = "rental_price_per_day"

X = df.drop(target_variable, axis=1)
Y = df.loc[:, target_variable]

# Automatically detect names of numeric/categorical columns
numeric_features = []
categorical_features = []
for i,t in X.dtypes.items():
    if ('float' in str(t)) or ('int' in str(t)) :
        numeric_features.append(i)
    else :
        categorical_features.append(i)

# Separate train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# StandardScaler to scale data (i.e apply Z-score) - OneHotEncoder to encode categorical variables
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

# Use ColumnTransformer to make a preprocessor object that describes all the treatments to be done
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

# Pipeline 
model = Pipeline(steps=[
    ("Preprocessing", preprocessor),
    ("Regressor",LinearRegression())
], verbose=True)


# Log experiment to MLFlow
with mlflow.start_run(run_id = run.info.run_id) as run:
    # Instanciate and fit the model 
    model.fit(X_train, Y_train)
    predictions = model.predict(X_train)
        
    # Log model seperately to have more flexibility on setup 
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="getaround_project",
        registered_model_name="api_linear_regression",
        signature=infer_signature(X_train, predictions)
    )

    # Print results 
    print("Done")