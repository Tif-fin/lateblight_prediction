import numpy as np
from geopy.distance import geodesic
from geopy.point import Point

def get_data(data):
    data = np.array(data)
    
    data = np.clip(data, 0, None)
    min_temperature = data[:, 0]  #3
    relative_humidity = data[:, 1]#1  
    precipitation = data[:, 2]#2 
    
    return min_temperature, relative_humidity, precipitation

def indexFunction(precipitaion_vec, min_temperature, relative_humidity):
    # Initialize variables
    D = np.zeros_like(precipitaion_vec)
    last_non_zero_index = -1

    # Iterate through the array
    for i in range(len(precipitaion_vec)):
        if precipitaion_vec[i] != 0:
            last_non_zero_index = i
        D[i] = i - last_non_zero_index
    #     D = D+1
    #     print(D-1)
    ####################################
    I = np.zeros_like(precipitaion_vec)
    for day in range(len(min_temperature)):
        if precipitaion_vec[day] == 0:
            idx = 100 + (min_temperature[day] - 10) + 2 * (relative_humidity[day] - 80) / D[day] + precipitaion_vec[day]
            I[day] = idx
        else:
            idx = 100 + (min_temperature[day] - 10) + 2 * (relative_humidity[day] - 80)
            I[day] = idx
    I_total = np.sum(I)
    return I_total

def wart_disease_chance(data):
    temperature,  humidity, precipitation = get_data(data=data)
    disease_chance = []
    if (12 <= np.mean(temperature) <= 24) and (np.sum(precipitation) > 700) and (np.mean(humidity) > 80):
        chance = np.mean(temperature) + (np.sum(precipitation) - 700) / 10 + np.mean(humidity) / 2
        # Normalize chance between 0 and 100
        normalized_chance = (chance - 12) / (24 - 12) * 100
        disease_chance.append(normalized_chance)
    else:
        disease_chance.append(0)  # If conditions not met, chance is 0

    return round(abs(np.mean(disease_chance)),2)

def bacterial_wilt_disease_chance(data):
    disease_chance = []
    
    temperature,  humidity, precipitation = get_data(data=data)

    if (30 <= np.mean(temperature) <= 35) and (np.mean(humidity) > 0):
        chance = np.mean(temperature) + np.mean(humidity)
        # Normalize chance between 0 and 100
        normalized_chance = (chance - 30) / (35 - 30) * 100
        disease_chance.append(normalized_chance)
    else:
        disease_chance.append(0)  # If conditions not met, chance is 0

    return round(abs(np.mean(disease_chance)),2)


def process_weather_data(data):
    data = np.array(data)
    
    data = np.clip(data, 0, None)

    min_temperature = data[:, 0]  #3
    relative_humidity = data[:, 1]#1  
    precipitation = data[:, 2]#2 

    return round(abs(indexFunction(precipitaion_vec = precipitation[-5:], min_temperature=min_temperature[-5:], relative_humidity=relative_humidity[-5:])/600), 2)



def geodesic_distance(lat1, lon1, lat2, lon2):
    # Create Point objects for the coordinates
    point1 = Point(latitude=lat1, longitude=lon1)
    point2 = Point(latitude=lat2, longitude=lon2)

    # Calculate the geodesic distance using Vincenty formula
    distance = geodesic(point1, point2).kilometers

    return distance