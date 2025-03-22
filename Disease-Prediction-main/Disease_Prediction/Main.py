import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

from scipy.stats import mode

import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv("Disease_Prediction/Training.csv").dropna(axis=1)

# print(data.shape,data_test.shape)
# print(data.head())

# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})
 
# plt.figure(figsize = (18,8))
# sns.barplot(x = "Disease", y = "Counts", data = temp_df,hue='Disease',palette='Spectral')#magma,spectral,coolwarm,viridis
# plt.xticks(rotation=90)
# plt.show()

# Converting the prognosis column to numerical values
le = LabelEncoder()
data['prognosis'] = le.fit_transform(data['prognosis'])
# print(data.head())

# Splitting the data
X = data.drop(columns=('prognosis')) # data.iloc[:,:-1]
y = data['prognosis'] # data.iloc[:,-1]
# print(X,y)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape, X_test.shape,
#        y_train.shape, y_test.shape)

# Defining scoring metric for KFold Validation
def cv_scoring(estimator,X,y):
    return accuracy_score(y, estimator.predict(X))

# Initializing Models
models = {'SVC':SVC(),
          'Gaussian NB':GaussianNB(),
          'Random Forest':RandomForestClassifier(random_state=42)}

# Producing Cross validation score of the models
for i in models:
    model = models[i]
    score = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring=cv_scoring)
    print(model)
    print(f"Scores : {score}")
    print(f"Mean : {np.mean(score)}")
    print("=="*50)

# Training and Testing SVM classifier
svm = SVC()
svm.fit(X_train,y_train)
preds = svm.predict(X_test)

print(f"Accuracy on train data by SVM Classifier = {accuracy_score(y_train,svm.predict(X_train))*100}")
print(f"Accuracy on test data by SVM Classifier = {accuracy_score(y_test,preds)*100}")
# cf_matrix = confusion_matrix(y_test,preds)
# plt.figure(figsize=(10,15))
# sns.heatmap(data=cf_matrix,annot=True)
# plt.title("Accuracy score of SVM on testing data")
# plt.show()
print("=="*50)

# Training and Testing Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train,y_train)
preds = nb.predict(X_test)
print(f"Accuracy on train data by Naive Bayes Classifier = {accuracy_score(y_train,nb.predict(X_train))*100}")
print(f"Accuracy on test data by Naive Bayes Classifier = {accuracy_score(y_test,preds)*100}")
# cf_matrix = confusion_matrix(y_test,preds)
# plt.figure(figsize=(10,15))
# sns.heatmap(data=cf_matrix,annot=True)
# plt.title("Accuracy score of Naive Bayes on testing data")
# plt.show()
print("=="*50)

# Training and Testing Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train,y_train)
preds = rfc.predict(X_test)
print(f"Accuracy on train data by Random Forest Classifier Classifier = {accuracy_score(y_train,rfc.predict(X_train))*100}")
print(f"Accuracy on test data by Random Forest Classifier Classifier = {accuracy_score(y_test,preds)*100}")
# cf_matrix = confusion_matrix(y_test,preds)
# plt.figure(figsize=(10,15))
# sns.heatmap(data=cf_matrix,annot=True)
# plt.title("Accuracy score of Random Forest Classifier on testing data")
# plt.show()
print("=="*50)

# Now Training the models on whole Training.csv and then Testing on whole Testing.csv
final_svm = SVC()
final_nb = GaussianNB()
final_rf = RandomForestClassifier()

final_svm.fit(X,y)
final_nb.fit(X,y)
final_rf.fit(X,y)

data_test = pd.read_csv("Disease_Prediction/Testing.csv").dropna(axis=1)
data_test['prognosis'] = le.transform(data_test['prognosis'])

test = data_test.iloc[:,:-1]
result = data_test.iloc[:,-1]
pred_svm = final_svm.predict(test)
pred_nb = final_nb.predict(test)
pred_rf = final_rf.predict(test)

# print(result.shape)
# print(result)
final_pred_mode = mode([pred_svm, pred_nb, pred_rf], axis=0)
final_pred = np.array(final_pred_mode[0])
# print(final_pred.shape,final_pred)
print(f"Accuracy on Test dataset by the combined model\
: {accuracy_score(result, final_pred)*100}")
 
# cf_matrix = confusion_matrix(result, final_pred)
# plt.figure(figsize=(12,8))
 
# sns.heatmap(cf_matrix, annot = True)
# plt.title("Confusion Matrix for Combined Model on Test Dataset")
# plt.show()

# Now Creating a function to predict disease based on given symptoms
symptoms = X.columns.values
 
# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

# print(symptom_index)

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":le.classes_
}
# print(data_dict)

# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
     
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
         
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
     
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm.predict(input_data)[0]]
     
    # making final prediction by taking mode of all predictions
    all_predictions = [rf_prediction, nb_prediction, svm_prediction]
    # final_prediction = max(set(all_predictions), key=all_predictions.count)
    unique_elements, counts = np.unique(all_predictions, return_counts=True)
    max_count_index = np.argmax(counts)
    final_prediction = unique_elements[max_count_index]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
    }
    return predictions

# Testing the function
print(predictDisease("Spotting  urination,Foul Smell Of urine,Continuous Feel Of Urine,Dark Urine"))