# streamlit run <filename>

import os
# import dog_breed_rfc
# from dog_breed_rfc import orig_dog_breeds
import joblib
import numpy as np
import pandas as pd
import random
from streamlit_extras.buy_me_a_coffee import button  # pip install streamlit-extras
from sklearn.metrics import accuracy_score, log_loss

#%%
try:
    import streamlit as st 
except ImportError:
    os.system('pip install streamlit')
    import streamlit as st

# Custom CSS
custom_css = """
<style>
    # USING ################################################

    /* Change the background color of the app */
    /* In first div under <div tabindex="-1"> */
    .stApp {
        # background-color: #003300;
        # background-color: #ffe6ff;
        background-color: #f2e6ff;
    }

    # NOT USING #############################################

    /* Change the font color of all streamlit text */
    * {
        # color: #ff6666;
        # color: #ff4d4d;
        # color: #ff1a1a;
        # color: #ffcc99;
        color: #ffffff;
        }

    /* changes the background color of the re-run ect developer bar at the top of the window which will not be shown after the application is deployed. Not useful for deployment. */
    /* In the first header within the above div */
    .st-emotion-cache-1avcm0n {
        # color: azure; /* very pale blue */
    }

    .st-emotion-cache-13k62yr {
        # color: #ff6699; /* changes the text color of the re-run ect text in the developer bar at the top of the window which will not be shown after the application is deployed, as well as the text paragraphs. Not useful for deployment. */

        # background: #ffe6ee; /* Not using. This can also change the background color like .stApp is doing */
    }
    
    /* changes color of first two paragraphs */
    /* .element-container { */
    .st-emotion-cache-uiblw7 e1f1d6gn4 {
        color: #ff6699;
    }

    /* changes text color of the submit button */
    .stButton {
        # color: #cc0044;
    }

    /* changes background color of the submit button */
    .st-emotion-cache-19rxjzo {
        # color: #cc0044;
        # background-color: #ff6699;
        # background-color: azure;
        # background-color: robinseggblue;
    }
</style>
"""

# function to run model with user input
# @st.cache_data
def predict_optimal_dog_breeds(model, size, num_dogs_had, home_size, alone_dogs, sensitivity, cold_weather, hot_weather, child_friendly, dog_friendly, stranger_friendly, shedding_amount, drooling_potential, groom_ease, weight_gain_potential, trainability, intelligence, mouthiness, prey_drive, barking, wanderlust_potential, en_lvl, intensity, ex_needs, playfulness):
    data = np.array([[num_dogs_had, home_size, alone_dogs, sensitivity, cold_weather, hot_weather, dog_friendly, stranger_friendly, shedding_amount, drooling_potential, groom_ease, weight_gain_potential, trainability, intelligence, mouthiness, prey_drive, barking, wanderlust_potential, en_lvl, intensity, ex_needs, playfulness,size,child_friendly]])
    # model = joblib.load('model.pkl')
    prediction = model.predict(data)
    return prediction

def random_choice_between(num1, num2):
    return random.choice([num1, num2])

# def display_breed_prediction_info(breed_prediction):
#     breeds_original = pd.read_csv('C:\\Users\\Linda\\OneDrive\\Desktop\\BYUI\\2024_Spring_Senior_Project\\dog_breed_classifier\\dataset\\breeds_original.csv')
#     breed_location = breeds_original['breed'] == str(breed_prediction)
#     breed_url = breeds_original.loc[breed_location, 'url']
#     return breed_url

def main():
    # Inject custom CSS with st.markdown function
    st.markdown(custom_css, unsafe_allow_html=True)

    from streamlit_extras.badges import badge  # pip install streamlit-extras

    # badge(type="buymeacoffee", name="CodingIsFun")

    with st.sidebar:
        badge(type="buymeacoffee", name="lindaspellman")
    ##########################################

    ############# MAIN SECTION
    min_slider_value = 1
    max_slider_value = 5
    default_slider_value = 3

    # Load the model
    model = joblib.load('orig_data_model.pkl')
    
    st.title('Dog Breed Advisor')
    st.subheader('By: Linda Spellman')
    st.write('Welcome to the Dog Breed Advisor! Here, would-be dog-owners can input details about their lifestyle conditions and receive recommendations of dog breeds that may fit with their lives.')
    st.write("My name is Linda Spellman. This Streamlit application constitutes my senior project as a Computer Science student at Brigham Young University - Idaho.")
    st.divider() 

    # size
    size = st.slider("1. On a scale of 1-5, what size dog do you want? 1. Tiny. 2. Small. 3. Medium. 4. Large. 5. Giant.", min_slider_value, max_slider_value, default_slider_value)
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
    home_size = st.slider("3. On a scale of 1-5, what size home do you live in? 1. Tiny Apartment (~400 square feet or less). 2. Apartment (~800 square feet). 3. Single-family home (~2000 square feet). 4. Large single-family home (~3500 square feet. 5. Mansion (at least 8000 square feet)", min_slider_value, max_slider_value, default_slider_value)
    # a4_tolerates_being_alone
    alone_dogs = st.slider("4. On a scale of 1-5, will you need to leave your dog alone for long periods of time?", min_slider_value, max_slider_value, default_slider_value)
    # a3_sensitivity_level
    sensitivity = st.slider("5. On a scale of 1-5, how sensitive a dog would you prefer? 1: Lowest sensitivity. 5: Highest sensitivity.", min_slider_value, max_slider_value, default_slider_value)
    # a5_tolerates_cold_weather
    cold_weather = st.slider("6. On a scale of 1-5, how cold is the coldest time of the year in your location?", min_slider_value, max_slider_value, default_slider_value)
    # a6_tolerates_hot_weather
    hot_weather = st.slider("7. On a scale of 1-5, how hot is the hottest time of the year in your location?",min_slider_value, max_slider_value, default_slider_value)
    # both_family_kid_friendliness
    child_friendly = st.slider("8. On a scale of 1-5, how often will your dog be around children? 1 = Never. 2 = Rarely. 3 = Probably. 4 = Often. 5 = I want a good family dog for my children.", min_slider_value, max_slider_value, default_slider_value)
    # b3_dog_friendly
    dog_friendly = st.slider("9. Will your dog need to be friendly towards other dogs or guard against strange dogs? 1 = Friendly. 5 = Guarded. ", min_slider_value, max_slider_value, default_slider_value)
    # b4_friendly_toward_strangers
    stranger_friendly = st.slider("10. In the same vein of thinking, will your dog need to guard against strangers or be friendly towards strangers?", min_slider_value, max_slider_value, default_slider_value)
    # c1_amount_of_shedding
    shedding_amount = st.slider("11. On a scale of 1-5, would you prefer a non-shedding dog, a sometimes-shedding dog, or are you willing to clean up after an often-shedding dog (if it is an indoor dog)?", min_slider_value, max_slider_value, default_slider_value)
    # c2_drooling_potential
    drooling_potential = st.slider("12. On a scale of 1-5, would you prfer a dog who doesn't drool or are you fine with drooling?",min_slider_value, max_slider_value, default_slider_value)
    # c3_easy_to_groom
    groom_ease = st.slider("13. On a scale of 1-5, would you prefer an easy-to-groom dog, are you planning to professionally groom your dog, or are you willing to groom a difficult-to-groom dog at home?", min_slider_value, max_slider_value, default_slider_value)
    # c5_potential_for_weight_gain
    weight_gain_potential = st.slider("14. Some dogs require feeding on a schedule or they will gain weight easily. On a scale of 1 to 5, would you prefer a dog who can regulate their eating schedule or are you disciplined enough to keep to a schedule for a dog who has a high potential for weight gain?", min_slider_value, max_slider_value, default_slider_value)
    # d1_easy_to_train
    trainability = st.slider("15. On a scale of 1-5, how trainable a dog would you prefer. 1: Challenging. 5: Easy to train.", min_slider_value, max_slider_value, default_slider_value)
    # d2_intelligence
    intelligence = st.slider("16. On a scale of 1-5, do you want a more intelligent dog, or are you fine with a less intelligent dog? 1: Dumb dog. 5: Very intelligent dog.", min_slider_value, max_slider_value, default_slider_value)
    # d3_potential_for_mouthiness
    mouthiness = st.slider("17. On a scale of 1-5, would you prefer a naturally non-mouthy dog or are you willing to put up with mouthiness/train your dog to not be mouthy?",min_slider_value, max_slider_value, default_slider_value)
    # d4_prey_drive
    prey_drive = st.slider("18. On a scale of 1-5, do you want a dog who is given to a high prey drive and will easily give chase, or do you want a dog who does not care? 1: Low prey drive. 5: High prey drive.", min_slider_value, max_slider_value, default_slider_value)
    # d5_tendency_to_bark_or_howl
    barking = st.slider("19. On a scale of 1-5, would you prefer a quiet dog or a loud dog? 1: Not prone to barking/howling. 5: Very prone to vocalizing.",min_slider_value, max_slider_value, default_slider_value)
    # d6_wanderlust_potential
    wanderlust_potential = st.slider("20. On a scale of 1-5, how comfortable with a dog having a high wanderlust potential? 1: Prefers little wanderlust. 5: Comfortable with higher wanderlust. ",min_slider_value, max_slider_value, default_slider_value)
    # e1_energy_level
    en_lvl = st.slider("21. On a scale of 1-5, do you need a high-energy dog or a low-energy dog? High-energy dogs are better suited to being working dogs, while low-energy dogs are often couch potatos. 1: Low-energy. 5: High-energy",min_slider_value, max_slider_value, default_slider_value)
    # e2_intensity
    intensity = st.slider("22. On a scale of 1-5, would you prefer a high-intensity dog or low-intensity dog? High-intensity dogs are energetic and enthusiastic in all their activities, while low-intensity dogs are more subdued and quiet. 1: Low-intensity. 5: High-intensity.", min_slider_value, max_slider_value, default_slider_value)
    # e3_exercise_needs
    ex_needs = st.slider("23. On a scale of 1-5, how willing are you to spend a lot of time walking your dog? Would you prefer short periods of exercise or are you willing to walk your dog for longer periods and further distances? 1: Prefer short walks. 5: Comfortable with long walks.", min_slider_value, max_slider_value, default_slider_value)
    # e4_potential_for_playfulness
    playfulness = st.slider("24. On a scale of 1-5, how interested are you in playing with your dog and how much time do you have to spend playing with your dog? Would you prefer a more playful dog or less playful dog? 1. Less playful. 5: More playful.", min_slider_value, max_slider_value, default_slider_value)

    # breed_recs = []
    # breed_urls = []
    breeds_original = pd.read_csv('C:\\Users\\Linda\\OneDrive\\Desktop\\BYUI\\2024_Spring_Senior_Project\\dog_breed_classifier\\dataset\\breeds_original.csv')

    preferences_submitted = st.button('Submit', key=1)
    if preferences_submitted:
        # st.balloons()
        st.write(f'Your dog breed recommendations are: ')
        # send all answers to the model
        breed_rec = predict_optimal_dog_breeds(model, size, num_dogs_had, home_size, alone_dogs, sensitivity, cold_weather, hot_weather, child_friendly, dog_friendly, stranger_friendly, shedding_amount, drooling_potential, groom_ease, weight_gain_potential, trainability, intelligence, mouthiness, prey_drive, barking, wanderlust_potential, en_lvl, intensity, ex_needs, playfulness)
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
        breed_rec2 = predict_optimal_dog_breeds(model, size
                                            +random_choice_between(-1,1)
                                            , num_dogs_had
                                            +random_choice_between(-1,1)
                                            , home_size
                                            +random_choice_between(-1,1)
                                            , alone_dogs+random_choice_between(-1,1)
                                            , sensitivity+random_choice_between(-1,1)
                                            , cold_weather+random_choice_between(-1,1)
                                            , hot_weather+random_choice_between(-1,1)
                                            , child_friendly+random_choice_between(-1,1)
                                            , dog_friendly+random_choice_between(-1,1)
                                            , stranger_friendly+random_choice_between(-1,1)
                                            , shedding_amount+random_choice_between(-1,1)
                                            , drooling_potential+random_choice_between(-1,1)
                                            , groom_ease+random_choice_between(-1,1)
                                            , weight_gain_potential+random_choice_between(-1,1)
                                            , trainability+random_choice_between(-1,1)
                                            , intelligence+random_choice_between(-1,1)
                                            , mouthiness+random_choice_between(-1,1)
                                            , prey_drive+random_choice_between(-1,1)
                                            , barking+random_choice_between(-1,1)
                                            , wanderlust_potential+random_choice_between(-1,1)
                                            , en_lvl+random_choice_between(-1,1)
                                            , intensity+random_choice_between(-1,1)
                                            , ex_needs+random_choice_between(-1,1)
                                            , playfulness+random_choice_between(-1,1))
        # st.write(breed_rec2)
        condition = breeds_original['breed'] == str(breed_rec2[0])
        breed_url2 = breeds_original.loc[condition, 'url']

        # third breed recommendation
        breed_rec3 = predict_optimal_dog_breeds(model, size
                                            +random_choice_between(-1,1)
                                            , num_dogs_had
                                            +random_choice_between(-1,1)
                                            , home_size
                                            +random_choice_between(-1,1)
                                            , alone_dogs+random_choice_between(-2,2)
                                            , sensitivity+random_choice_between(-2,2)
                                            , cold_weather+random_choice_between(-2,2)
                                            , hot_weather+random_choice_between(-2,2)
                                            , child_friendly+random_choice_between(-2,2)
                                            , dog_friendly+random_choice_between(-2,2)
                                            , stranger_friendly+random_choice_between(-2,2)
                                            , shedding_amount+random_choice_between(-2,2)
                                            , drooling_potential+random_choice_between(-2,2)
                                            , groom_ease+random_choice_between(-2,2)
                                            , weight_gain_potential+random_choice_between(-2,2)
                                            , trainability+random_choice_between(-2,2)
                                            , intelligence+random_choice_between(-2,2)
                                            , mouthiness+random_choice_between(-2,2)
                                            , prey_drive+random_choice_between(-2,2)
                                            , barking+random_choice_between(-2,2)
                                            , wanderlust_potential+random_choice_between(-2,2)
                                            , en_lvl+random_choice_between(-2,2)
                                            , intensity+random_choice_between(-2,2)
                                            , ex_needs+random_choice_between(-2,2)
                                            , playfulness+random_choice_between(-2,2))
        condition = breeds_original['breed'] == str(breed_rec3[0])
        breed_url3 = breeds_original.loc[condition, 'url']

        # fourth breed recommendation
        breed_rec4 = predict_optimal_dog_breeds(model, size
                                            +random_choice_between(-1,1)
                                            , num_dogs_had
                                            +random_choice_between(-1,1)
                                            , home_size
                                            +random_choice_between(-1,1)
                                            , alone_dogs+random_choice_between(-3,3)
                                            , sensitivity+random_choice_between(-3,3)
                                            , cold_weather+random_choice_between(-3,3)
                                            , hot_weather+random_choice_between(-3,3)
                                            , child_friendly+random_choice_between(-3,3)
                                            , dog_friendly+random_choice_between(-3,3)
                                            , stranger_friendly+random_choice_between(-3,3)
                                            , shedding_amount+random_choice_between(-3,3)
                                            , drooling_potential+random_choice_between(-3,3)
                                            , groom_ease+random_choice_between(-3,3)
                                            , weight_gain_potential+random_choice_between(-3,3)
                                            , trainability+random_choice_between(-3,3)
                                            , intelligence+random_choice_between(-3,3)
                                            , mouthiness+random_choice_between(-3,3)
                                            , prey_drive+random_choice_between(-3,3)
                                            , barking+random_choice_between(-3,3)
                                            , wanderlust_potential+random_choice_between(-3,3)
                                            , en_lvl+random_choice_between(-3,3)
                                            , intensity+random_choice_between(-3,3)
                                            , ex_needs+random_choice_between(-3,3)
                                            , playfulness+random_choice_between(-3,3))
        condition = breeds_original['breed'] == str(breed_rec4[0])
        breed_url4 = breeds_original.loc[condition, 'url']

        # fifth breed recommendation
        breed_rec5 = predict_optimal_dog_breeds(model, size
                                            +random_choice_between(-1,1)
                                            , num_dogs_had
                                            +random_choice_between(-1,1)
                                            , home_size
                                            +random_choice_between(-1,1)
                                            , alone_dogs+random_choice_between(-4,4)
                                            , sensitivity+random_choice_between(-4,4)
                                            , cold_weather+random_choice_between(-4,4)
                                            , hot_weather+random_choice_between(-4,4)
                                            , child_friendly+random_choice_between(-4,4)
                                            , dog_friendly+random_choice_between(-4,4)
                                            , stranger_friendly+random_choice_between(-4,4)
                                            , shedding_amount+random_choice_between(-4,4)
                                            , drooling_potential+random_choice_between(-4,4)
                                            , groom_ease+random_choice_between(-4,4)
                                            , weight_gain_potential+random_choice_between(-4,4)
                                            , trainability+random_choice_between(-4,4)
                                            , intelligence+random_choice_between(-4,4)
                                            , mouthiness+random_choice_between(-4,4)
                                            , prey_drive+random_choice_between(-4,4)
                                            , barking+random_choice_between(-4,4)
                                            , wanderlust_potential+random_choice_between(-4,4)
                                            , en_lvl+random_choice_between(-4,4)
                                            , intensity+random_choice_between(-4,4)
                                            , ex_needs+random_choice_between(-4,4)
                                            , playfulness+random_choice_between(-4,4))
        condition = breeds_original['breed'] == str(breed_rec5[0])
        breed_url5 = breeds_original.loc[condition, 'url']

        rec_url1 = pd.DataFrame({'breed': [breed_rec[0]], 'url': breed_url})
        st.dataframe(rec_url1, hide_index=True)
        rec_url2 = pd.DataFrame({'breed': [breed_rec2[0]], 'url': breed_url2})
        st.dataframe(rec_url2, hide_index=True)
        rec_url3 = pd.DataFrame({'breed': [breed_rec3[0]], 'url': breed_url3})
        st.dataframe(rec_url3, hide_index=True)
        rec_url4 = pd.DataFrame({'breed': [breed_rec4[0]], 'url': breed_url4})
        st.dataframe(rec_url4, hide_index=True)
        rec_url5 = pd.DataFrame({'breed': [breed_rec5[0]], 'url': breed_url5})
        st.dataframe(rec_url5, hide_index=True)
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


            # TODO: replace lists with dictionary using .fromkeys() method?
            # TODO: write get_url() function to get the url from the breed name?

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

        st.write("Find information about these breeds at the above urls. Machine learning models can be wrong, so please research the breeds as much as possible before adopting. Please adopt, don't shop!")
        # if any of the recs in the list are the same, rerun the model. Basically, I want a set of unique recs.
        st.write("If you like this application, please consider buying me a coffee. Thank you!")

        model_ref = model['model']
        accuracies = model['accuracies']
        losses = model['losses']

        epochs = range(1, len(accuracies) + 1)
        metrics_data = pd.DataFrame({
            'Epoch': epochs,
            'Accuracy': accuracies,
            'Loss': losses
        })

        # Generate some example validation data
        X_val = np.random.rand(20, 10)
        y_val = np.random.randint(0, 2, 20)

        # Make predictions
        y_pred = model_ref.predict(X_val)
        y_pred_proba = model_ref.predict_proba(X_val)

        # Calculate accuracy and loss
        accuracy = accuracy_score(y_val, y_pred)
        loss = log_loss(y_val, y_pred_proba)

        # Streamlit app
        st.title('RandomForest Model Evaluation')

        # Display results
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Loss: {loss:.2f}')

        # Line chart for Accuracy
        st.subheader('Accuracy over Epochs')
        st.line_chart(metrics_data[['Epoch', 'Accuracy']].set_index('Epoch'))

        # Line chart for Loss
        st.subheader('Loss over Epochs')
        st.line_chart(metrics_data[['Epoch', 'Loss']].set_index('Epoch'))
        
if __name__ == "__main__":
    main()