'''
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score, confusion_matrix ,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', palette = 'deep', rc = {'axes.grid':True})
#%matplotlib inline



data = pd.read_csv('D:\\FYP\\Datasets\\survey lung cancer.csv')
data.head()

data_new = data.drop(['GENDER','AGE', 'SMOKING', 'ALCOHOL CONSUMING', 'CHRONIC DISEASE', 'PEER_PRESSURE', 'ALLERGY '], axis = 1)
symptoms = [ 'YELLOW_FINGERS', 'ANXIETY', 'FATIGUE ',  'WHEEZING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
X = data_new[symptoms]
y = data_new.LUNG_CANCER
X.head()



X_train, X_test, y_train, y_test = train_test_split( X, y, random_state = 42, stratify = y)


X_train.isnull().sum()


#X.info()


le = LabelEncoder()
y_train= le.fit_transform(y_train)
y_test= le.transform(y_test)




key = {2: 'yes', 1: 'no'}
for sys in symptoms:
	sns.countplot(x = X_train[sys].replace(key))    
	#plt.show()



sns.set(style = 'darkgrid',palette = 'bright')
sns.countplot(x = pd.Series(y_train).replace([0,1],['No','Yes']))



model =  RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#print('The accuracy score of this Random Forest Classifier model is: {0:.1f}%'.format(100*accuracy_score(y_test, y_pred)))




#features importance
Symptoms_importance = pd.DataFrame(  {"Symptoms": list(X.columns), "importance": model.feature_importances_}).sort_values("importance", ascending=False)
# Display
print(Symptoms_importance)
'''


'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load data from Excel file
data = pd.read_csv('D:\\FYP\\Datasets\\survey lung cancer.csv')

# Features (X) and target variable (y)
features = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE',
            'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
X = data[features]
y = data['LUNG_CANCER']

# Encode categorical features
le_gender = LabelEncoder()
X['GENDER'] = le_gender.fit_transform(X['GENDER'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode the target variable
le_lung_cancer = LabelEncoder()
y_train = le_lung_cancer.fit_transform(y_train)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
model.fit(X_train, y_train)

user_input = {}
# Function to take user input for symptoms and make a prediction
def predict_lung_cancer():
    
    for feature in features:
        if feature == 'GENDER':
            response = input(f"Enter your {feature} (M/F): ").upper()
            user_input[feature] = le_gender.transform([response])[0]
        elif feature == 'AGE':
            response = int(input(f"Enter your {feature}: "))
            user_input[feature] = response
        else:
            response = input(f"Do you have {feature}? (1 for No, 2 for Yes): ")
            user_input[feature] = int(response)

    # Convert user input to a DataFrame
    user_data = pd.DataFrame([user_input])

    # Make prediction
    prediction = model.predict(user_data)[0]

    # Decode the prediction
    prediction_label = le_lung_cancer.inverse_transform([prediction])[0]

    print(f"\nBased on the provided information, the model predicts: {prediction_label}")

# Make predictions based on user input
predict_lung_cancer()



#Important Output to check the Model.

# Print feature importances
print("Feature Importances:")
print(pd.Series(model.feature_importances_, index=features))


from sklearn.metrics import classification_report

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# Evaluate training accuracy
train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy}")


from sklearn.metrics import classification_report

# Convert string labels to integers in y_true
y_true = y_test.map({'NO': 0, 'YES': 1})

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_true, y_pred))




# Display feature importance
print("Feature importance:")
print(pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False))

# Assuming 'features' is the list of feature names
user_data = pd.DataFrame([user_input], columns=features)

print("User Input:")
print(user_data)

# Make prediction
prediction = model.predict(user_data)[0]
print("\nModel Prediction:", prediction)

# Decode the prediction
prediction_label = le_lung_cancer.inverse_transform([prediction])[0]
print("\nDecoded Prediction:", prediction_label)







'''

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from Excel file
data = pd.read_csv('C:/Users/Hassan-PC/Desktop/LCD/survey lung cancer.csv')

# Features (X) and target variable (y)
features = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE',
            'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
X = data[features]
y = data['LUNG_CANCER']

# Encode categorical features
le_gender = LabelEncoder()
X['GENDER'] = le_gender.fit_transform(X['GENDER'])

# Encode the target variable
le_lung_cancer = LabelEncoder()
y = le_lung_cancer.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
#joblib.dump(model, 'lung_cancer_model.joblib')
#print("Model saved successfully.")


# Function to take user input for symptoms and make a prediction
def predict_lung_cancer():
    user_input = {}
    for feature in features:
        if feature == 'GENDER':
            response = input(f"Enter your {feature} (M/F): ").upper()
            user_input[feature] = le_gender.transform([response])[0]
        elif feature == 'AGE':
            response = int(input(f"Enter your {feature}: "))
            user_input[feature] = response
        else:
            response = input(f"Do you have {feature}? (1 for No, 2 for Yes): ")
            user_input[feature] = int(response)

    # Convert user input to a DataFrame
    user_data = pd.DataFrame([user_input])

    # Make prediction
    regression_output = model.predict(user_data)[0]
    re=regression_output

    # Convert regression output to binary classification
    prediction_label = 'YES' if regression_output > 0.5 else 'NO'

    print(f"\nBased on the provided information, the model predicts: {prediction_label}")
    #print(re)

# Make predictions based on user input
#predict_lung_cancer()


coefficients = model.coef_
feature_names = X_train.columns

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})

# Display the importance DataFrame
#print(importance_df)



#print(regression_output)




from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predictions on the test set
y_pred = model.predict(X_test)

# Regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')











