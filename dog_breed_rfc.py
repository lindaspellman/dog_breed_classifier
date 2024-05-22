# %%
# RandomForest classifier - small amount of columns - need better accuracy than random guessing
# transformer
# SVM - support vector machine 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

#%%
dog_breeds = pd.read_csv('C:\\Users\\Linda\\OneDrive\\Desktop\\BYUI\\2024_Spring_Senior_Project\\dog_breed_classifier\\dataset\\breeds.csv')

dog_breeds['both_family_kid_friendliness'] = dog_breeds[['b1_affectionate_with_family','b2_incredibly_kid_friendly_dogs']].mean(axis=1)

#%%
# dog_breeds.head()

##########################################################################################

#%%
dog_breed_features = np.array(dog_breeds)

num_breeds = int(len(dog_breeds) / 2)
breed_labels = {} 
for i in range(num_breeds):
    column = [0] * (num_breeds * 2)
    column[i] = 1 
    column[i + num_breeds] = 1
    breed_labels[i] = column
    
breed_labels = pd.DataFrame(breed_labels)

#######################################################################################


# %%
X = dog_breeds[['a_adaptability'
                ,'a1_adapts_well_to_apartment_living'
                ,'a2_good_for_novice_owners'
                ,'a3_sensitivity_level'
                ,'a4_tolerates_being_alone'
                # ,'a5_tolerates_cold_weather'
                # ,'a6_tolerates_hot_weather'
                # ,'b3_dog_friendly'
                # ,'b4_friendly_toward_strangers'
                # ,'c_health_grooming'
                # ,'c1_amount_of_shedding'
                # ,'c2_drooling_potential'
                # ,'c3_easy_to_groom'
                # ,'c4_general_health'
                # ,'c5_potential_for_weight_gain'
                # ,'d_trainability'
                # ,'d1_easy_to_train'
                # ,'d2_intelligence'
                # ,'d3_potential_for_mouthiness'
                # ,'d4_prey_drive'
                # ,'d5_tendency_to_bark_or_howl'
                # ,'d6_wanderlust_potential'
                # ,'e_exercise_needs','e1_energy_level'
                # ,'e2_intensity'
                # ,'e3_exercise_needs'
                # ,'e4_potential_for_playfulness'
                # ,'both_family_kid_friendliness'
                # ,'size'
                ]]

#%%
# y = breed_labels
y = dog_breeds['breed']

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)




#%%
# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=256, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)




# %%
# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)



# %%
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')



#%%
# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))



#%%
# Print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# %%
