import requests 
import json
import pandas as pd 

#### Test ML Model 

def test_prediction():
    df = pd.read_csv('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv', index_col=0)
    
    columns = ['private_parking_available', 'has_gps','has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']
    df[columns] = df[columns].astype('object')
    
    for col in columns:
        df[col] = df[col].map(lambda x: "yes" if x else "no")
        
    df = df.sample(1)
    values = []
    for element in df.iloc[0,:].values.tolist():
        if type(element) != str:
            values.append(element.item())
        else:
            values.append(element)
    df_dict = {key:value for key, value in zip(df.columns, values)}

    r = requests.post(
        "https://ceribou-api-7104afe788e2.herokuapp.com/predict",
        data=json.dumps(df_dict)
    )

    response = r
    print(response)
    print(response.json())

test_prediction()

### Test batch pred
def test_batch():
    import csv
    import urllib3

    df = pd.read_csv('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv', index_col=0)
    
    columns = ['private_parking_available', 'has_gps','has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']
    df[columns] = df[columns].astype('object')
    
    for col in columns:
        df[col] = df[col].map(lambda x: "yes" if x else "no")
    
    df = df.sample(5)
    myfile = df.to_csv()
    r = requests.post(
        "https://ceribou-api-7104afe788e2.herokuapp.com/batch-predict",
        files={"file":myfile}
    )

    response = r
    print(response)
    print(response.json())

test_batch()

#### Prepare test data 
def prepare_test_file():

    df = pd.read_csv('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv', index_col=0)
    
    columns = ['private_parking_available', 'has_gps','has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']
    df[columns] = df[columns].astype('object')
    
    for col in columns:
        df[col] = df[col].map(lambda x: "yes" if x else "no")
        
    target_col_name= "rental_price_per_day"
    df = df.loc[:, df.columns != target_col_name]
    df = df.sample(20)
    df.to_csv("data/test_data.csv", index=False)
    return "Done"

prepare_test_file()