# Import required libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle

# Keras specific
import tensorflow
import keras
from keras.models import load_model

import datetime
import holidays

# ================================================


# ================================================
def model(data):

    # print(data)
    dir = "models/"
    # load the model
    model = load_model(dir+'env_model_3.h5')
    maxes = pickle.load(open(dir+'maxes.pkl', 'rb'))
    medians = pickle.load(open(dir+'medians.pkl', 'rb'))
    # ================================================

    # ================================================
    def convert_date(date,property):
        property[date_fields] = 0
        day = "Day_" + date.strftime("%A")
        month = "Month_" + str(date.month)
        year = "Year_" + str(date.year)
        
        property[[day,month,year]] = 1
        
        if date in us_holidays:
            property["is_holiday"] = 1
        return property

    def normalize_data(data):
        nonlocal maxes, medians
        if not data['host_is_superhost']:
            data['host_is_superhost'] = 0

        for column in ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'review_scores_cleanliness', 'review_scores_rating',
                    'review_scores_location', 'review_scores_value', 'price']:
            if data[column] or data[column] == 0:
                data[column] /= maxes[column]
            else:
                data[column] = medians[column]

        if not data['Neighbourhood']:
            neighbourhood = "Brooklyn"
            data['Neighbourhood'] = neighbourhood

        neighbourhood = "Neighbourhood_"+data['Neighbourhood']
        data['Neighbourhood'] = neighbourhood
        
        return data

    def create_property(data):
        row = {'host_is_superhost': data['host_is_superhost'], 'accommodates': data['accommodates'], 'bathrooms': data['bathrooms'], 
        'bedrooms': data['bedrooms'], 'beds': data['beds'], 'review_scores_cleanliness': data['review_scores_cleanliness'], 
        'review_scores_rating': data['review_scores_rating'], 'review_scores_location': data['review_scores_location'],
        'review_scores_value' : data['review_scores_value'], 'price': data['price'],
        'Neighbourhood_Bronx': 0, 'Neighbourhood_Brooklyn': 0,'Neighbourhood_Manhattan': 0, 
        'Neighbourhood_Queens': 0, 'Neighbourhood_Staten Island': 0, 'is_holiday': 0, 'Day_Friday': 0, 'Day_Monday': 0,
        'Day_Saturday': 0, 'Day_Sunday': 0, 'Day_Thursday': 0, 'Day_Tuesday': 0,
        'Day_Wednesday': 0, 'Month_1': 0, 'Month_2': 0, 'Month_3': 0, 'Month_4': 0, 'Month_5': 0,
        'Month_6': 0, 'Month_7': 0, 'Month_8': 0, 'Month_9': 0, 'Month_10': 0, 'Month_11': 0,
        'Month_12': 0, 'Year_2021': 0, 'Year_2022': 0, 'Year_2023': 0}
        property = pd.DataFrame([row])
        property[data["Neighbourhood"]] = 1
        
        return property
    # ================================================

    # ================================================
    us_holidays = holidays.US()
    date_fields = ['is_holiday', 'Day_Friday', 'Day_Monday',
        'Day_Saturday', 'Day_Sunday', 'Day_Thursday', 'Day_Tuesday',
        'Day_Wednesday', 'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5',
        'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11',
        'Month_12', 'Year_2021', 'Year_2022', 'Year_2023']
    date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2024, 1, 1)
    delta = datetime.timedelta(days=1)

    data = normalize_data(data)
    property = create_property(data)

    # print(property.drop(date_fields,axis=1))

    # MAKE PROPERTY
    properties = pd.DataFrame(property)

    while date < end_date:
        property = convert_date(date,property)
        properties = pd.concat([properties, property], ignore_index=True)
        date += delta

    # PREDICT PROPERTY
    res = model.predict(properties, verbose=0)
    res = res[1:]

    res = res.flatten().tolist()

    # plt.plot(np.arange(1, 366), res)
    # plt.gcf().set_figwidth(15)
    # plt.show()

    return res
    # ================================================

# data = {'host_is_superhost' : 0, 'accommodates' : None, 'bathrooms' : None, 'bedrooms' : 2, 'beds' : 4,
#         'review_scores_cleanliness' : 5, 'review_scores_rating' : 4,
#         'review_scores_location' : 4, 'review_scores_value' : 5, 'price' : None,
#         'Neighbourhood' : 'Manhattan'}

# print(model(data))