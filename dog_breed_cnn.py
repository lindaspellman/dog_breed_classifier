# !!!!!!!!!!!!!!!!
# DONE - size col to combine approx. height and weight - tiny, small, medium, large, giant (for user) - 1 through 5 (for neural network)
# !!!!!!!!!!!!!!!!
# research how/where tensorflow models can output predictions associated with truth. Find the file which lists the ordinal values for each image in image classifiers
#%%


#%%
import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
#%%
dog_breeds = pd.read_csv('C:\\Users\\Linda\\OneDrive\\Desktop\\BYUI\\2024_Spring_Senior_Project\\dog_breed_classifier\\dataset\\breeds.csv')

# average family and kid friendliness columns
dog_breeds['both_family_kid_friendliness'] = dog_breeds[['b1_affectionate_with_family','b2_incredibly_kid_friendly_dogs']].mean(axis=1)

# List of columns to delete
columns_to_delete = ['index', 'breed', 'url','b_all_around_friendliness','b1_affectionate_with_family','b2_incredibly_kid_friendly_dogs','c6_size','breed_group', 'height','weight','life_span','height_range_in'] # consider getting rid of 'b_all_around_friendliness' if it doesn't converge

# Delete the specified columns
dog_breeds.drop(columns=columns_to_delete, inplace=True)

dog_breeds.head()

#%%
dog_breeds.columns

#%%
# unique_heights = 
# len(dog_breeds['height'].unique())
# unique_heights

#%%
# Define a function to apply the case when logic
def apply_case_when(value):
    if value == 'condition1':
        return 'result1'
    elif value == 'condition2':
        return 'result2'
    else:
        return 'default_result'

# Apply the function to a specific column in your DataFrame
dog_breeds['size'] = dog_breeds['height_in'].apply(apply_case_when)
# large range --> create two diff sizes/rows of the breed

# combine family and kid friendliness vs unfriendly with unfamiliar people vs. friendly with everyone

# test a much smaller part of the dataset - 35 inputs?

# !!!!!!!!!!!!
# manually assign values to parse height, weight, and life_span
# constrain weights by height picked first
# research actually teacup vs regular size of dog breeds

# Data Wrangling: regularizing data for training
# get rid of all_around_friendliness and breed_group columns
# split height, weight, and life_span columns into multiple rows so that the data will converge and make sense
# change ranges into ordinals for drop down boxes- 10-15 --> 10-12 and 13-15, 9-12 --> 10-12
# 70-140 lbs --> 75-100 and 100-125 - tell accurate number in web page result
# associate ordinals/standard ranges with numbers 1,2,3,...
# assign dogs to closest range
# HAVE A WORKING MODEL THIS WEEK TO NEXT WEEK

##############################################

# The CNN

