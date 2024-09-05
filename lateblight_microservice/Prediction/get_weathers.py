import pandas as pd
from datetime import datetime, timedelta
import requests
import csv
from io import StringIO

def get_three_days_back_date():
    current_date = datetime.now()
    three_days_back_date = current_date - timedelta(days=3)
    formatted_date = three_days_back_date.strftime('%Y%m%d')
    return formatted_date

def get_forty_seven_days_back_date():
    current_date = datetime.now()
    forty_four_days_back_date = current_date - timedelta(days=124)
    formatted_date = forty_four_days_back_date.strftime('%Y%m%d')
    return formatted_date

def make_api_call(latitude, longitude, place_name):
    #'T2M', 'RH2M', 'PRECTOTCORR', 'T2MWET', 'TS', 'WS10M_MAX'
    url = f'https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,PRECTOT,PS,QV2M,RH2M,T2MWET,T2M_MAX,T2M_MIN,T2M_RANGE,TS,WS10M,WS10M_MAX,WS10M_MIN,WS10M_RANGE,WS50M,WS50M_MAX,WS50M_MIN,WS50M_RANGE&community=SB&longitude={longitude}&latitude={latitude}&start={get_forty_seven_days_back_date()}&end={get_three_days_back_date()}&format=CSV'
    response = requests.get(url)

    if response.status_code == 200:
        content = response.text
        data = content.split("-END HEADER-")[1]

        # Convert string to CSV format
        csv_data = StringIO(data)
        csv_reader = csv.reader(csv_data)

        # Skip the first row with headers
        next(csv_reader)

        # Append data to the list
        csv_file = list(csv_reader)
        headers = csv_file[0]

        data = csv_file[1:]

        df = pd.DataFrame(data, columns=headers)
        df['Longitude'] = longitude
        df['Latitude'] = latitude
        df['Location'] = place_name

        # Combine 'YEAR', 'MO', and 'DY' columns into a new 'Date' column separated by '-'
        df['Date'] = df['YEAR'].astype(str) + '-' + df['MO'].astype(str) + '-' + df['DY'].astype(str)

        # Drop the original 'YEAR', 'MO', 'DY' columns
        df.drop(columns=['YEAR', 'MO', 'DY'], inplace=True)

        # Reorder the columns
        cols = ['Date', 'Location', 'Latitude', 'Longitude', 'T2M', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']
        df = df[cols]

        return df
    else:
        return None

def process_locations_and_return_csv(locations_file_path):
    df_locations = pd.read_csv(locations_file_path)

    compiled_csv_data = pd.DataFrame()

    prev_longitude = ""
    prev_latitude = ""

    for index, row in df_locations.iterrows():
        place = row['Location']
        latitude = row['Latitude']
        longitude = row['Longitude']

        if prev_latitude == latitude and prev_longitude == longitude:
            pass
        else:
            print(f"Getting weather data of {place}")
            api_data = make_api_call(latitude, longitude, place)

        prev_longitude = longitude
        prev_latitude = latitude

        if api_data is not None:
            compiled_csv_data = pd.concat([compiled_csv_data, api_data])

    print("test data prepared")
    data_path = 'static/Test_Data/test_data.csv'
    compiled_csv_data.to_csv(data_path, index=False)
    return data_path


