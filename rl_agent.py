# Import required libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle

import gym
from gym import spaces
import random

# Keras specific
import tensorflow
import keras
from keras.models import load_model
import time

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

import datetime
import holidays

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# LOAD
# =================================================================
from models.airbnb_env import Airbnb
# load the model
filename = 'env_model_3.h5'
dir = "models/"
# model = pickle.load(open(filename, 'rb'))
model = load_model(dir+filename)

X_train = pickle.load(open(dir+'X_train.pkl', 'rb'))
X_test = pickle.load(open(dir+'X_test.pkl', 'rb'))
y_train = pickle.load(open(dir+'y_train.pkl', 'rb'))
maxes = pickle.load(open(dir+'maxes.pkl', 'rb'))
medians = pickle.load(open(dir+'medians.pkl', 'rb'))
maxes_price = maxes["price"]
# =================================================================


# MODEL SETUP
# =================================================================
def make_env():
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = Airbnb(X_train, y_train, model)
        return env
    return _init

env = Airbnb(X_train, y_train, model)

# wrap it
env = make_vec_env(make_env(), n_envs=20)

save_dir = dir+"RL/"
RLFile = "/RLModel"
# LOAD
RLModel = PPO.load(save_dir + RLFile, verbose=1)
RLModel.set_env(env)

# =================================================================



# FUNCTIONS
# =================================================================

def plot_avg(mean_rewards):
    mean_rewards = np.array(mean_rewards)
    plt.figure(figsize=(15, 10))  
    plt.plot(mean_rewards)
    # Add a title and labels to the x and y axes
    plt.title("Rewards Average per Evaluation")
    plt.xlabel("Evaluation")
    plt.ylabel("Rewards Average")

    # Display the plot
    plt.show()

def getReward(state):
    price = state['price'].values[0]
    bookingConfidence = model.predict(state, verbose=0)[0][0]
    if bookingConfidence > 0.85: return price
    if bookingConfidence < 0.15: return 0

    chance = random.uniform(0, 1)
    if chance < bookingConfidence: return (price + (price*bookingConfidence))/2
    return (0 + (price*bookingConfidence))/2

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

# =================================================================



# INITIALIZE
# =================================================================
us_holidays = holidays.US()
date_fields = ['is_holiday', 'Day_Friday', 'Day_Monday',
       'Day_Saturday', 'Day_Sunday', 'Day_Thursday', 'Day_Tuesday',
       'Day_Wednesday', 'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5',
       'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11',
       'Month_12', 'Year_2021', 'Year_2022', 'Year_2023']
delta = datetime.timedelta(days=1)

# =================================================================


# DYNAMIC
# =================================================================
def dynamic_pricing(data):
    test_rewards = []
    test_actions = []
    week_rewards = []
    week_actions = []
    daily_rewards = []
    daily_actions = []
    action_predictions = []

    date = datetime.date(2023, 1, 1)

    data = normalize_data(data)
    property = create_property(data)

    start_time = time.time() 

    for i in range(365):

        property = convert_date(date, property)

        # create weeks
        if property["Day_Monday"].item() == 1:
            test_rewards.append(np.mean(week_rewards))
            test_actions.append(np.mean(week_actions))
            week_rewards = []
            week_actions = []

        # actions
        property_modified = [property.to_numpy().astype(np.float16)]

        action_predictions = []
        for j in range(10):
            action, _states = RLModel.predict(property_modified)
            action_predictions.append(action[0][0][0])
        test_action = np.mean(action_predictions)
        property['price'] = test_action

        #reward
        reward = getReward(property)
        week_rewards.append(reward)
        week_actions.append(test_action)
        daily_rewards.append(reward)
        daily_actions.append(test_action)

        # update for next
        date += delta

    result_actions = np.array(test_actions) * maxes_price
    result_rewards = np.array(test_rewards) * maxes_price
    result_daily_rewards = np.array(daily_rewards) * maxes_price
    result_daily_actions = np.array(daily_actions) * maxes_price
    total = np.sum(result_daily_rewards)

    end_time = time.time()
    print("Time taken:", end_time - start_time)
    return result_actions.tolist(), result_rewards.tolist(), round(total, 3), result_daily_rewards, result_daily_actions

# data = {'host_is_superhost' : 0, 'accommodates' : None, 'bathrooms' : None, 'bedrooms' : 2, 'beds' : 4,
#        'review_scores_cleanliness' : 5, 'review_scores_rating' : 4,
#        'review_scores_location' : 4, 'review_scores_value' : 5, 'price' : None,
#        'Neighbourhood' : 'Manhattan'}

# result_actions, result_rewards, total, result_daily_rewards, result_daily_actions =  dynamic_pricing(data)
# print("TOTAL DYNAMIC:", total, result_actions)

# =================================================================


# STATIC
# =================================================================
def static_pricing(data):

    test_rewards = []
    week_rewards = []
    daily_rewards = []

    date = datetime.date(2022, 1, 1)

    data = normalize_data(data)
    property = create_property(data)
    set_price = property["price"]

    start_time = time.time() 

    for i in range(365):
        property = convert_date(date, property)

        # create weeks
        if property["Day_Monday"].item() == 1:
            test_rewards.append(np.mean(week_rewards))
            week_rewards = []

        #reward
        reward = getReward(property)
        week_rewards.append(reward)
        daily_rewards.append(reward)

        # update for next
        date += delta

    result_rewards_static = np.array(test_rewards) * maxes_price
    daily_rewards_static = np.array(daily_rewards) * maxes_price
    total_static = np.sum(daily_rewards_static)
    
    end_time = time.time()
    print("Time taken:", end_time - start_time)
    return result_rewards_static.tolist(), round(total_static, 3), set_price

# data = {'host_is_superhost' : 0, 'accommodates' : None, 'bathrooms' : None, 'bedrooms' : 2, 'beds' : 4,
#        'review_scores_cleanliness' : 5, 'review_scores_rating' : 4,
#        'review_scores_location' : 4, 'review_scores_value' : 5, 'price' : None,
#        'Neighbourhood' : 'Manhattan'}
# result_rewards_static, total_static, set_price = static_pricing(data)
# print("TOTAL STATIC:", total_static)
# =================================================================

# # PLOT 1
# # =================================================================
def plot1():
    plt.title("Daily Recommendation")
    plt.plot(np.arange(1, len(result_daily_actions)+1),result_daily_actions)
    plt.plot(np.arange(1, len(result_daily_rewards)+1),result_daily_rewards, color='red')
    plt.axhline(y=set_price.item()*maxes_price, color='grey', linestyle='-')
    plt.gcf().set_figwidth(15)
    plt.show()

    plt.title("Weekly Recommendation")
    plt.plot(np.arange(1, len(result_actions)+1),result_actions)
    plt.plot(np.arange(1, len(result_rewards)+1),result_rewards, color='r')
    plt.axhline(y=set_price.item()*maxes_price, color='grey', linestyle='-')
    plt.gcf().set_figwidth(15)
    plt.show()
# # =================================================================

# # PLOT 2
# # =================================================================
def plot2():
    import seaborn as sns
    import calendar

    # Set the seaborn style and color palette
    sns.set_style('darkgrid')
    sns.color_palette("deep")

    # Define the figure size and dpi for the plot
    fig, ax = plt.subplots(figsize=(15,6), dpi=100)

    # Plot the data and fill the area under the curves
    ax.plot(np.arange(1, len(result_rewards)+1), result_rewards, label='Dynamic Price Revenue', linewidth=3, color=sns.color_palette("husl", 9)[4])
    ax.plot(np.arange(1, len(result_rewards_static)+1), result_rewards_static, label='Fixed Price Revenue', linewidth=3, color=sns.color_palette("deep")[3])
    ax.fill_between(np.arange(1, len(result_rewards)+1), result_rewards, alpha=0.1,color=sns.color_palette("husl", 9)[4])
    ax.fill_between(np.arange(1, len(result_rewards_static)+1), result_rewards_static, alpha=0.1, color=sns.color_palette("deep")[3])

    # Add a horizontal line for the original price
    ax.axhline(y=set_price.item()*maxes_price, color=sns.color_palette("deep")[1],linestyle='-', label='Original Price', linewidth=3)

    # Set the title and axis labels
    ax.set_title('RL Dynamic Revenue vs Fixed Price Revenue', fontsize=20)
    ax.set_xlabel('Weeks - 2023', fontsize=16)
    ax.set_ylabel('Price - ($)', fontsize=16)

    # Set the x and y-axis tick label font size
    ax.tick_params(axis='both', labelsize=12)
    
    # Set the x-axis limits
    ax.set_xlim([0, len(result_rewards)+1])

    # Add a legend
    ax.legend(fontsize=14, loc='lower right')

    # Show the plot
    plt.show()
# =================================================================

# plot1()
# plot2()
