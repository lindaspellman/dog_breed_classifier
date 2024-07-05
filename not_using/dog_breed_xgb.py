#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score

#%%
# Load your dataset
dog_breeds = pd.read_csv('C:\\Users\\Linda\\OneDrive\\Desktop\\BYUI\\2024_Spring_Senior_Project\\dog_breed_classifier\\dataset\\breeds.csv')
dog_breeds['both_family_kid_friendliness'] = dog_breeds[['b1_affectionate_with_family','b2_incredibly_kid_friendly_dogs']].mean(axis=1)

# Assume the last column is the target (dog breed) and others are features
# X = dog_breeds.iloc[:, :-1].values
X = dog_breeds[[#'a_adaptability'
                # ,
                'a1_adapts_well_to_apartment_living'
                ,'a2_good_for_novice_owners'
                ,'a3_sensitivity_level'
                ,'a4_tolerates_being_alone'
                ,'a5_tolerates_cold_weather'
                ,'a6_tolerates_hot_weather'
                ,'b3_dog_friendly'
                ,'b4_friendly_toward_strangers'
                # ,'c_health_grooming'
                ,'c1_amount_of_shedding'
                ,'c2_drooling_potential'
                ,'c3_easy_to_groom'
                ,'c4_general_health'
                ,'c5_potential_for_weight_gain'
                #,'d_trainability'
                ,'d1_easy_to_train'
                ,'d2_intelligence'
                ,'d3_potential_for_mouthiness'
                ,'d4_prey_drive'
                ,'d5_tendency_to_bark_or_howl'
                ,'d6_wanderlust_potential'
                # ,'e_exercise_needs'
                ,'e1_energy_level'
                ,'e2_intensity'
                ,'e3_exercise_needs'
                ,'e4_potential_for_playfulness'
                ,'both_family_kid_friendliness'
                ,'size'
                ]]
# y = dog_breeds.iloc[:, -1].values
y = dog_breeds['breed']


# Encode the target labels to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

### Step 3: Train the XGBoost Model
# Now, train the XGBoost classifier on the training data:

# Convert the datasets into DMatrix objects, which is an optimized data structure for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up the parameters for XGBoost
params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': len(label_encoder.classes_),  # Number of classes
    'max_depth': 6,  # Maximum depth of a tree
    'eta': 0.3,  # Learning rate
    'eval_metric': 'mlogloss'  # Evaluation metric
}

# Train the model
num_rounds = 100  # Number of boosting rounds
bst = xgb.train(params, dtrain, num_rounds)

### Step 4: Make Predictions and Evaluate the Model
# Make predictions on the test set and evaluate the model's performance:

# Make predictions
preds = bst.predict(dtest)

# Calculate accuracy
accuracy = accuracy_score(y_test, preds)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Optionally, you can decode the predicted labels back to the original breed names
predicted_breeds = label_encoder.inverse_transform(preds.astype(int))
#%%

### Step 5: (Optional) Save and Load the Model
# You might want to save the trained model for later use:
# Save the model
bst.save_model('xgboost_dog_breeds.model')

#%%
# Load the model
bst_loaded = xgb.Booster()
bst_loaded.load_model('xgboost_dog_breeds.model')

# Use the loaded model to make predictions
preds_loaded = bst_loaded.predict(dtest)
# %%
len(preds_loaded)
# %%
dog_breeds['predictions'] = preds_loaded
dog_breeds.head() # LAST CELL RUN FOR THE ERROR

# %%