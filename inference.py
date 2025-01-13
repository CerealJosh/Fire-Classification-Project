import joblib
import tensorflow as tf
import numpy as np
import warnings

data_path = 'C:/Users/Josh/source/repos/ProjectServer/Josh-main/data.json'
labels_path = 'C:/Users/Josh/source/repos/ProjectServer/Josh-main/labels.txt'
labels = []

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
with open(labels_path,'r') as f:
    for f in f.readlines():
        labels.append(f.split('\t')[1].strip())
with open(data_path, 'r') as g:
        Z = g.read().strip()
        Z= Z.strip("[]")
        Z = Z.split(",")

files_list = ['C:/Users/Josh/source/repos/ProjectServer/Josh-main/logistic_regression.pkl',
              'C:/Users/Josh/source/repos/ProjectServer/Josh-main/decision_tree.pkl',
              'C:/Users/Josh/source/repos/ProjectServer/Josh-main/k_neighbors.pkl',
              'C:/Users/Josh/source/repos/ProjectServer/Josh-main/random_forest.pkl',
              'C:/Users/Josh/source/repos/ProjectServer/Josh-main/gradient_boost.pkl',
              'C:/Users/Josh/source/repos/ProjectServer/Josh-main/x_gradient_boost.pkl',
              'C:/Users/Josh/source/repos/ProjectServer/Josh-main/stacking.pkl',
              'C:/Users/Josh/source/repos/ProjectServer/Josh-main/neural_network.keras']
def get_predictions(model_num,features):
    
    features = np.reshape(features,[1,5])
    if model_num in range(1,8):
        model = joblib.load(files_list[model_num-1])
        pred = model.predict(features)

    elif model_num == 8:
        model = tf.keras.models.load_model(files_list[model_num-1])
        pred = tf.argmax(model.predict(features),-1)
        
    return labels[pred[0]]


features = []
feature_list = ['MQ-5','MQ-135','MQ-2','MQ-3','TD']
for i in range(5):
    #features.append(Z[i])
    features.append(float(Z[i]))

# model_num = int(input(f'Which model to use......\n\
#                 1.Logistic Regression\n\
#                 2.Decision Tree\n\
#                 3.KNeighbors\n\
#                 4.Random Forest\n\
#                 5.Gradient Boosting\n\
#                 6.Extreme Gradient Boosting\n\
#                 7.Stacking\n\
#                 8.Neural Network\n'))

model_num = 4

print(get_predictions(model_num,features))
