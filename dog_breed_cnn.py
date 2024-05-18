# !!!!!!!!!!!!!!!!
# DONE - size col to combine approx. height and weight - tiny, small, medium, large, giant (for user) - 1 through 5 (for neural network)
# !!!!!!!!!!!!!!!!
# research how/where tensorflow models can output predictions associated with truth. Find the file which lists the ordinal values for each image in image classifiers
#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
# Import the libraries we need
# from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

#%%
dog_breeds = pd.read_csv('C:\\Users\\Linda\\OneDrive\\Desktop\\BYUI\\2024_Spring_Senior_Project\\dog_breed_classifier\\dataset\\breeds.csv')

# average family and kid friendliness columns
dog_breeds['both_family_kid_friendliness'] = dog_breeds[['b1_affectionate_with_family','b2_incredibly_kid_friendly_dogs']].mean(axis=1)

# List of columns to delete
columns_to_delete = ['url','b_all_around_friendliness','b1_affectionate_with_family','b2_incredibly_kid_friendly_dogs','c6_size','breed_group', 'height','weight','life_span','height_range_in'] # consider getting rid of 'b_all_around_friendliness' if it doesn't converge

# Delete the specified columns
dog_breeds.drop(columns=columns_to_delete, inplace=True)

# check the data to make sure it's right


#%%
dog_breeds.head()

#%%
column_count = len(dog_breeds.columns)
column_count

#%%
no_headers = pd.read_csv('C:\\Users\\Linda\\OneDrive\\Desktop\\BYUI\\2024_Spring_Senior_Project\\dog_breed_classifier\\dataset\\breeds.csv', header=None)
# Drop the first row
no_headers = no_headers.drop(0)
# Reset the index if needed
no_headers = no_headers.reset_index(drop=True)
no_headers = no_headers.drop(no_headers.columns[[0,1,2,34,35,36,37,38]], axis=1)
no_headers
# figure out how to display no_headers without the numbered column names

dog_breed_features = np.array(no_headers)

#%%
num_rows = len(no_headers)
breed_labels = {} 
for i in range(num_rows): 
    row = [0] * num_rows 
    row[i] = 1 
    breed_labels[i] = row 

#%% 
# Convert all non-integer values to integers
# breeds.iloc[:, :] = breeds.apply(pd.to_numeric, errors='coerce', downcast='integer')

# # Write the modified DataFrame to a new CSV file with headers
# breeds.to_csv('dataset\\dog_breeds.csv', index=False)

#%%
# unique_heights = 
# len(dog_breeds['height'].unique())
# unique_heights

#%%
# Define a function to apply the case when logic
# def apply_case_when(value):
#     if value == 'condition1':
#         return 'result1'
#     elif value == 'condition2':
#         return 'result2'
#     else:
#         return 'default_result'

# Apply the function to a specific column in your DataFrame
# dog_breeds['size'] = dog_breeds['height_in'].apply(apply_case_when)
# large range --> create two diff sizes/rows of the breed

# combine family and kid friendliness vs unfriendly with unfamiliar people vs. friendly with everyone

# test a much smaller part of the dataset - 35 inputs?

# Data Wrangling: regularizing data for training
# get rid of all_around_friendliness and breed_group columns
# split height, weight, and life_span columns into multiple rows so that the data will converge and make sense
# change ranges into ordinals for drop down boxes- 10-15 --> 10-12 and 13-15, 9-12 --> 10-12
# 70-140 lbs --> 75-100 and 100-125 - tell accurate number in web page result
# associate ordinals/standard ranges with numbers 1,2,3,...
# assign dogs to closest range

##############################################
#%%
# The CNN
# dog_breeds = pd.read_csv('C:\\Users\\Linda\\OneDrive\\Desktop\\BYUI\\2024_Spring_Senior_Project\\dog_breed_classifier\\dataset\\dog_breeds.csv')

# Get our target variable and features and split them into test and train datasets
X = dog_breeds[['a_adaptability'
                ,'a1_adapts_well_to_apartment_living'
                , 'a2_good_for_novice_owners'
                , 'a3_sensitivity_level'
                , 'a4_tolerates_being_alone'
                ,'a5_tolerates_cold_weather'
                ,'a6_tolerates_hot_weather'
                ,'b3_dog_friendly'
                ,'b4_friendly_toward_strangers'
                ,'c_health_grooming'
                ,'c1_amount_of_shedding'
                ,'c2_drooling_potential'
                ,'c3_easy_to_groom'
                ,'c4_general_health'
                ,'c5_potential_for_weight_gain'
                ,'d_trainability'
                ,'d1_easy_to_train'
                ,'d2_intelligence'
                ,'d3_potential_for_mouthiness'
                ,'d4_prey_drive'
                ,'d5_tendency_to_bark_or_howl'
                ,'d6_wanderlust_potential'
                ,'e_exercise_needs','e1_energy_level'
                ,'e2_intensity'
                ,'e3_exercise_needs'
                ,'e4_potential_for_playfulness'
                ,'both_family_kid_friendliness'
                ,'size'
                ]]
y = dog_breeds['index']

from sklearn.preprocessing import StandardScaler

# prob with data: only one example of each dog breed. Given testing data random inputs for dog breeds to be returned on?
# need only one friendly column, ect, so it won't overwhelm size column. Or weight size more?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# how do you define how many inputs and outputs you're using??? Specify my training data - count number of rows, columns
# hidden layers - more neurons?

# %%
# Create the model and train it, use default hyperparameters for now
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'), 
    # input_shape=(X_train_scaled.shape[dog_breed_features.shape[1],1]),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(dog_breed_features.shape[0], activation='sigmoid') # does sigmoid give good linear results?
    ])
print('Model Defined')

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model Compiled")

# Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=358, validation_split=0.2)
model.fit(dog_breed_features, breed_labels, epochs=10)

# model = XGBRegressor()
# model.fit(X_train, y_train)
print("Model Fitted")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Make predictions
predictions = model.predict(X_test)
print("Predictions Made")

# %%
# print("Predictions Printed: ")
# predictions
print("Predicted Index:")
for prediction in predictions:
    print(prediction)

# %%


