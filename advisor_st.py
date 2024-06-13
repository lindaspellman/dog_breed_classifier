# streamlit run <filename>

import os
# import dog_breed_rfc
# from dog_breed_rfc import orig_dog_breeds
import joblib
import numpy as np
import pandas as pd
import random

#%%
try:
    import streamlit as st 
except ImportError:
    os.system('pip install streamlit')
    import streamlit as st

# Load the model
model = joblib.load('model.pkl')

# function to run model with user input
def predict_optimal_dog_breeds(size, num_dogs_had, home_size, alone_dogs, sensitivity, cold_weather, hot_weather, child_friendly, dog_friendly, stranger_friendly, shedding_amount, drooling_potential, groom_ease, weight_gain_potential, trainability, intelligence, mouthiness, prey_drive, barking, wanderlust_potential, en_lvl, intensity, ex_needs, playfulness):
    data = np.array([[size, num_dogs_had, home_size, alone_dogs, sensitivity, cold_weather, hot_weather, child_friendly, dog_friendly, stranger_friendly, shedding_amount, drooling_potential, groom_ease, weight_gain_potential, trainability, intelligence, mouthiness, prey_drive, barking, wanderlust_potential, en_lvl, intensity, ex_needs, playfulness]])
    prediction = model.predict(data)
    return prediction

def random_positive_or_negative(min_val, max_val):
    # Generate a random integer between min_val and max_val
    number = random.randint(min_val, max_val)
    # Randomly choose a sign (either 1 or -1)
    sign = random.choice([1, -1])
    # Combine the number and the sign
    return number * sign

# def display_breed_prediction_info(breed_prediction):
#     breeds_original = pd.read_csv('C:\\Users\\Linda\\OneDrive\\Desktop\\BYUI\\2024_Spring_Senior_Project\\dog_breed_classifier\\dataset\\breeds_original.csv')
#     breed_location = breeds_original['breed'] == str(breed_prediction)
#     breed_url = breeds_original.loc[breed_location, 'url']
#     return breed_url
    
st.title('Dog Breed Advisor')
st.write('Welcome to the Dog Breed Advisor! Here, would-be dog-owners can input details about their lifestyle conditions and receive recommendations of dog breeds that may fit with their lives.')
st.write("My name is Linda Spellman. This website's functioning constitutes my senior project as Computer Science student at Brigham Young University - Idaho. The machine learning system uses RandomForestClassifier.")
st.divider() 

# size
size = st.slider("1. On a scale of 1-5, what size dog do you want? 1. Tiny. 2. Small. 3. Medium. 4. Large. 5. Giant.",1,5,3)
# a2_good_for_novice_owners
num_dogs_had = st.slider('2. How many dogs have you had experienced ownership with by this point in your life? 0, 1, 2, 3, or 4+?', 0, 4, 2)
if num_dogs_had == 0:
    num_dogs_had = 1
elif num_dogs_had == 1:
    num_dogs_had = 2
elif num_dogs_had == 2:
    num_dogs_had = 3
elif num_dogs_had == 3:
    num_dogs_had = 4
elif num_dogs_had == 4:
    num_dogs_had = 5
# a1_adapts_well_to_apartment_living
home_size = st.slider("3. On a scale of 1-5, what size home do you live in? 1. Tiny Apartment (~400 square feet or less). 2. Apartment (~800 square feet). 3. Single-family home (~2000 square feet). 4. Large single-family home (~3500 square feet. 5. Mansion (at least 8000 square feet)", 1, 5, 3)
# a4_tolerates_being_alone
alone_dogs = st.slider("4. On a scale of 1-5, will you need to leave your dog alone for long periods of time?", 1, 5, 3)
# a3_sensitivity_level
sensitivity = st.slider("5. On a scale of 1-5, how sensitive a dog would you prefer? 1: Lowest sensitivity. 5: Highest sensitivity.", 1,5,3)
# a5_tolerates_cold_weather
cold_weather = st.slider("6. On a scale of 1-5, how cold is the coldest time of the year in your location?", 1, 5, 3)
# a6_tolerates_hot_weather
hot_weather = st.slider("7. On a scale of 1-5, how hot is the hottest time of the year in your location?",1,5,3)
# both_family_kid_friendliness
child_friendly = st.slider("8. On a scale of 1-5, how often will your dog be around children? 1 = Never. 2 = Rarely. 3 = Probably. 4 = Often. 5 = I want a good family dog for my children.", 1, 5, 3)
# b3_dog_friendly
dog_friendly = st.slider("9. Will your dog need to be friendly towards other dogs or guard against strange dogs? 1 = Friendly. 5 = Guarded. ", 1, 5, 3)
# b4_friendly_toward_strangers
stranger_friendly = st.slider("10. In the same vein of thinking, will your dog need to guard against strangers or be friendly towards strangers?", 1, 5, 3)
# c1_amount_of_shedding
shedding_amount = st.slider("11. On a scale of 1-5, would you prefer a non-shedding dog, a sometimes-shedding dog, or are you willing to clean up after an often-shedding dog (if it is an indoor dog)?", 1, 5, 3)
# c2_drooling_potential
drooling_potential = st.slider("12. On a scale of 1-5, would you prfer a dog who doesn't drool or are you fine with drooling?",1,5,3)
# c3_easy_to_groom
groom_ease = st.slider("13. On a scale of 1-5, would you prefer an easy-to-groom dog, are you planning to professionally groom your dog, or are you willing to groom a difficult-to-groom dog at home?", 1, 5, 3)
# c5_potential_for_weight_gain
weight_gain_potential = st.slider("14. Some dogs require feeding on a schedule or they will gain weight easily. On a scale of 1 to 5, would you prefer a dog who can regulate their eating schedule or are you disciplined enough to keep to a schedule for a dog who has a high potential for weight gain?", 1, 5, 3)
# d1_easy_to_train
trainability = st.slider("15. On a scale of 1-5, how trainable a dog would you prefer. 1: Challenging. 5: Easy to train.", 1, 5, 3)
# d2_intelligence
intelligence = st.slider("16. On a scale of 1-5, do you want a more intelligent dog, or are you fine with a less intelligent dog? 1: Dumb dog. 5: Very intelligent dog.", 1, 5,3)
# d3_potential_for_mouthiness
mouthiness = st.slider("17. On a scale of 1-5, would you prefer a naturally non-mouthy dog or are you willing to put up with mouthiness/train your dog to not be mouthy?",1,5,3)
# d4_prey_drive
prey_drive = st.slider("18. On a scale of 1-5, do you want a dog who is given to a high prey drive and will easily give chase, or do you want a dog who does not care? 1: Low prey drive. 5: High prey drive.", 1, 5, 3)
# d5_tendency_to_bark_or_howl
barking = st.slider("19. On a scale of 1-5, would you prefer a quiet dog or a loud dog? 1: Not prone to barking/howling. 5: Very prone to vocalizing.",1,5,3)
# d6_wanderlust_potential
wanderlust_potential = st.slider("20. On a scale of 1-5, how comfortable with a dog having a high wanderlust potential? 1: Prefers little wanderlust. 5: Comfortable with higher wanderlust. ", 1,5,3)
# e1_energy_level
en_lvl = st.slider("21. On a scale of 1-5, do you need a high-energy dog or a low-energy dog? High-energy dogs are better suited to being working dogs, while low-energy dogs are often couch potatos. 1: Low-energy. 5: High-energy",1,5,3)
# e2_intensity
intensity = st.slider("22. On a scale of 1-5, would you prefer a high-intensity dog or low-intensity dog? High-intensity dogs are energetic and enthusiastic in all their activities, while low-intensity dogs are more subdued and quiet. 1: Low-intensity. 5: High-intensity.", 1, 5, 3)
# e3_exercise_needs
ex_needs = st.slider("23. On a scale of 1-5, how willing are you to spend a lot of time walking your dog? Would you prefer short periods of exercise or are you willing to walk your dog for longer periods and further distances? 1: Prefer short walks. 5: Comfortable with long walks.", 1,5,3)
# e4_potential_for_playfulness
playfulness = st.slider("24. On a scale of 1-5, how interested are you in playing with your dog and how much time do you have to spend playing with your dog? Would you prefer a more playful dog or less playful dog? 1. Less playful. 5: More playful.",1,5,3)

breed_recs = []
breed_urls = []
breeds_original = pd.read_csv('C:\\Users\\Linda\\OneDrive\\Desktop\\BYUI\\2024_Spring_Senior_Project\\dog_breed_classifier\\dataset\\breeds_original.csv')

submitted = st.button('Submit', key=1)
if submitted:
    st.balloons()
    st.write(f'Your dog breed recommendation is: ')
    # send all answers to the model
    breed_rec = predict_optimal_dog_breeds(size, num_dogs_had, home_size, alone_dogs, sensitivity, cold_weather, hot_weather, child_friendly, dog_friendly, stranger_friendly, shedding_amount, drooling_potential, groom_ease, weight_gain_potential, trainability, intelligence, mouthiness, prey_drive, barking, wanderlust_potential, en_lvl, intensity, ex_needs, playfulness)
    # st.write(breed_rec)
    # breed_recs.append(breed_rec)
    # for rec in breed_recs:
        # st.write(rec)
    condition = breeds_original['breed'] == str(breed_rec[0])
    breed_url = breeds_original.loc[condition, 'url']
    
    # st.write(breed_url)
        # breed_urls.append(breed_url)
    # for url in breed_urls:
        # st.write(url)
    # st.write(breed_rec)
    
    # st.write(breed_url)

    # for i in range(1,5):
    breed_rec2 = predict_optimal_dog_breeds(size
                                        #    +random_positive_or_negative(-1,1)
                                        , num_dogs_had
                                        #    +random_positive_or_negative(-1,1)
                                        , home_size
                                        # +random_positive_or_negative(-1,1)
                                        , alone_dogs+random_positive_or_negative(-1,1)
                                        , sensitivity+random_positive_or_negative(-1,1)
                                        , cold_weather+random_positive_or_negative(-1,1)
                                        , hot_weather+random_positive_or_negative(-1,1)
                                        , child_friendly+random_positive_or_negative(-1,1)
                                        , dog_friendly+random_positive_or_negative(-1,1)
                                        , stranger_friendly+random_positive_or_negative(-1,1)
                                        , shedding_amount+random_positive_or_negative(-1,1)
                                        , drooling_potential+random_positive_or_negative(-1,1)
                                        , groom_ease+random_positive_or_negative(-1,1)
                                        , weight_gain_potential+random_positive_or_negative(-1,1)
                                        , trainability+random_positive_or_negative(-1,1)
                                        , intelligence+random_positive_or_negative(-1,1)
                                        , mouthiness+random_positive_or_negative(-1,1)
                                        , prey_drive+random_positive_or_negative(-1,1)
                                        , barking+random_positive_or_negative(-1,1)
                                        , wanderlust_potential+random_positive_or_negative(-1,1)
                                        , en_lvl+random_positive_or_negative(-1,1)
                                        , intensity+random_positive_or_negative(-1,1)
                                        , ex_needs+random_positive_or_negative(-1,1)
                                        , playfulness+random_positive_or_negative(-1,1))
    # st.write(breed_rec2)
    condition = breeds_original['breed'] == str(breed_rec2[0])
    breed_url2 = breeds_original.loc[condition, 'url']

    # third breed recommendation
    breed_rec3 = predict_optimal_dog_breeds(size
                                        #    +random_positive_or_negative(-1,1)
                                        , num_dogs_had
                                        #    +random_positive_or_negative(-1,1)
                                        , home_size
                                        # +random_positive_or_negative(-1,1)
                                        , alone_dogs+random_positive_or_negative(-1,1)
                                        , sensitivity+random_positive_or_negative(-1,1)
                                        , cold_weather+random_positive_or_negative(-1,1)
                                        , hot_weather+random_positive_or_negative(-1,1)
                                        , child_friendly+random_positive_or_negative(-1,1)
                                        , dog_friendly+random_positive_or_negative(-1,1)
                                        , stranger_friendly+random_positive_or_negative(-1,1)
                                        , shedding_amount+random_positive_or_negative(-1,1)
                                        , drooling_potential+random_positive_or_negative(-1,1)
                                        , groom_ease+random_positive_or_negative(-1,1)
                                        , weight_gain_potential+random_positive_or_negative(-1,1)
                                        , trainability+random_positive_or_negative(-1,1)
                                        , intelligence+random_positive_or_negative(-1,1)
                                        , mouthiness+random_positive_or_negative(-1,1)
                                        , prey_drive+random_positive_or_negative(-1,1)
                                        , barking+random_positive_or_negative(-1,1)
                                        , wanderlust_potential+random_positive_or_negative(-1,1)
                                        , en_lvl+random_positive_or_negative(-1,1)
                                        , intensity+random_positive_or_negative(-1,1)
                                        , ex_needs+random_positive_or_negative(-1,1)
                                        , playfulness+random_positive_or_negative(-1,1))
    condition = breeds_original['breed'] == str(breed_rec3[0])
    breed_url3 = breeds_original.loc[condition, 'url']

    # fourth breed recommendation
    breed_rec4 = predict_optimal_dog_breeds(size
                                        #    +random_positive_or_negative(-1,1)
                                        , num_dogs_had
                                        #    +random_positive_or_negative(-1,1)
                                        , home_size
                                        # +random_positive_or_negative(-1,1)
                                        , alone_dogs+random_positive_or_negative(-1,1)
                                        , sensitivity+random_positive_or_negative(-1,1)
                                        , cold_weather+random_positive_or_negative(-1,1)
                                        , hot_weather+random_positive_or_negative(-1,1)
                                        , child_friendly+random_positive_or_negative(-1,1)
                                        , dog_friendly+random_positive_or_negative(-1,1)
                                        , stranger_friendly+random_positive_or_negative(-1,1)
                                        , shedding_amount+random_positive_or_negative(-1,1)
                                        , drooling_potential+random_positive_or_negative(-1,1)
                                        , groom_ease+random_positive_or_negative(-1,1)
                                        , weight_gain_potential+random_positive_or_negative(-1,1)
                                        , trainability+random_positive_or_negative(-1,1)
                                        , intelligence+random_positive_or_negative(-1,1)
                                        , mouthiness+random_positive_or_negative(-1,1)
                                        , prey_drive+random_positive_or_negative(-1,1)
                                        , barking+random_positive_or_negative(-1,1)
                                        , wanderlust_potential+random_positive_or_negative(-1,1)
                                        , en_lvl+random_positive_or_negative(-1,1)
                                        , intensity+random_positive_or_negative(-1,1)
                                        , ex_needs+random_positive_or_negative(-1,1)
                                        , playfulness+random_positive_or_negative(-1,1))
    condition = breeds_original['breed'] == str(breed_rec4[0])
    breed_url4 = breeds_original.loc[condition, 'url']

    # fifth breed recommendation
    breed_rec5 = predict_optimal_dog_breeds(size
                                        #    +random_positive_or_negative(-1,1)
                                        , num_dogs_had
                                        #    +random_positive_or_negative(-1,1)
                                        , home_size
                                        # +random_positive_or_negative(-1,1)
                                        , alone_dogs+random_positive_or_negative(-1,1)
                                        , sensitivity+random_positive_or_negative(-1,1)
                                        , cold_weather+random_positive_or_negative(-1,1)
                                        , hot_weather+random_positive_or_negative(-1,1)
                                        , child_friendly+random_positive_or_negative(-1,1)
                                        , dog_friendly+random_positive_or_negative(-1,1)
                                        , stranger_friendly+random_positive_or_negative(-1,1)
                                        , shedding_amount+random_positive_or_negative(-1,1)
                                        , drooling_potential+random_positive_or_negative(-1,1)
                                        , groom_ease+random_positive_or_negative(-1,1)
                                        , weight_gain_potential+random_positive_or_negative(-1,1)
                                        , trainability+random_positive_or_negative(-1,1)
                                        , intelligence+random_positive_or_negative(-1,1)
                                        , mouthiness+random_positive_or_negative(-1,1)
                                        , prey_drive+random_positive_or_negative(-1,1)
                                        , barking+random_positive_or_negative(-1,1)
                                        , wanderlust_potential+random_positive_or_negative(-1,1)
                                        , en_lvl+random_positive_or_negative(-1,1)
                                        , intensity+random_positive_or_negative(-1,1)
                                        , ex_needs+random_positive_or_negative(-1,1)
                                        , playfulness+random_positive_or_negative(-1,1))
    condition = breeds_original['breed'] == str(breed_rec5[0])
    breed_url5 = breeds_original.loc[condition, 'url']


    st.write('1. ' + breed_rec + ' - ' + breed_url + ' ')
    st.write('2. ' + breed_rec2 + ' - ' + breed_url2 + ' ')
    st.write('3. ' + breed_rec3 + ' - ' + breed_url3 + ' ')
    st.write('4. ' + breed_rec4 + ' - ' + breed_url4 + ' ')
    st.write('5. ' + breed_rec5 + ' - ' + breed_url5 + ' ')
        # breed_recs.append(breed_rec)
        ###################
        # st.write(breed_rec)
        # breed_location = breeds_original['breed'] == str(breed_rec)
        # breed_url = breeds_original.loc[breed_location, 'url']
        # st.write(breed_url)
##########################

        # breed_dict = dict.fromkeys(f'{breed_recs}')
        # breed_dict[i] = breed_url
        # breed_urls.append(breed_url)


        # TO DO: replace lists with dictionary using .fromkeys() method?
        # TO DO: write get_url() function to get the url from the breed name?

    # st.write(breed_recs)
    # st.write(breed_urls)
    # st.write(breed_dict)

    # loop through breed_recs and print out the breed name and url

    # for index, rec in enumerate(breed_recs):
    # for index, rec in enumerate(breed_dict):
    # for rec in breed_rec:
        # breed_url = display_breed_prediction_info(rec)
        # st.write(f'{index+1}. {rec}')
        # for url in breed_urls:
            # st.write(url)

        # st.write(breed_url)

    st.write("Find information about this breed at the above urls. Machine learning models can be wrong, so please research the breed recommendation as much as possible before adopting.")
    # if any of the recs in the list are the same, rerun the model. Basically, I want a set of unique recs.
    