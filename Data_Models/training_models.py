# %% [markdown]
# 
# ### **Importing libraries for loading the data,visualization and eda**

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# %% [markdown]
# Getting all the feature variables and the scenario label

# %%
data = pd.read_csv('FireProject.csv',usecols=['MQ-5','MQ-135','MQ-2','MQ-3','TD','SCENARIO'])

# %% [markdown]
# Getting the statistics on each feature column

# %%
data.describe()

# %% [markdown]
# Encoding scenario label column

# %%
data['scenario_label_cat'] = data['SCENARIO'].astype('category')
y=  data['SCENARIO']
data['scenario_label_cat'] = data['scenario_label_cat'].cat.codes
y_one_hot = pd.get_dummies(y).values
print(y_one_hot)

# %% [markdown]
# Plotting the correlation matrix heatmap to check relationship between feature variables and target variable

# %%
temp = data
corr_matrix = temp.drop('SCENARIO',axis=1).corr()
plt.figure(figsize=(8,4))
sn.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# ### **Importing sklearn for the baseline models:**
# 
# A logistic regression classifier\
# A decision tree classifier\
# A kneighbors classifier

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier

# %% [markdown]
# Splitting the dataset into training and testing splits with 80:20 ratio

# %%
from sklearn.model_selection import train_test_split
X = data.drop(['scenario_label_cat','SCENARIO'],axis=1)
y = data['scenario_label_cat']
X_train, X_test, y_train, y_test = train_test_split(X,y_one_hot,test_size=.2,random_state=21)

# %% [markdown]
# Training a logistic regression model with a combination of l1 and l2 regularization/penalty

# %%
log_reg = LogisticRegression(solver='saga',penalty='elasticnet',max_iter=500,l1_ratio=.5)
log_reg.fit(X_train,y_train)

# %% [markdown]
# Testing the trained logistic regression model on the testing dataset

# %%
log_reg.score(X_test,y_test)

# %% [markdown]
# Training a decision tree model

# %%
tree_clfr = DecisionTreeClassifier()
tree_clfr.fit(X_train,y_train)

# %% [markdown]
# Testing the trained decision tree model on the testing dataset

# %%
tree_clfr.score(X_test,y_test)

# %% [markdown]
# Training a k-neighbors model using 8 neighbors

# %%
knn_clfr = KNeighborsClassifier(n_neighbors=8)
knn_clfr.fit(X_train,y_train)

# %% [markdown]
# Testing the trained k-neighbors model on the testing dataset

# %%
knn_clfr.score(X_test,y_test)

# %% [markdown]
# ### **Importing tensorflow for the neural network model**

# %%
import tensorflow as tf;
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics
from tensorflow.keras.utils import to_categorical

# %% [markdown]
# Using a simple neural network with 3 fully connected inner layers\
# with 64,128, and 256 units, all with relu activation\
# The output layer is a fully connected layer with softmax activation 

# %%
n_classes = data['SCENARIO'].nunique()

nn_clfr = Sequential([
    Dense(64,activation='relu',input_shape=[5]),
    Dense(128,activation='relu',),
    Dense(256,activation='relu'),
    Dense(n_classes,activation='softmax')]
)

# %% [markdown]
# Getting a summary of the neural network's architecture

# %%
nn_clfr.summary()

# %% [markdown]
# Training the neural network for 25 iterations/epochs\
# using sparse categorical cross entropy loss\
# and the Adaptive Moment Estimation (Adam) optimizer

# %%
nn_clfr.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy',metrics.Precision(), metrics.Recall()]
)

nn_history = nn_clfr.fit(X_train,y_train,epochs=25,batch_size=64,validation_data=[X_test,y_test])

# %%
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(nn_history.history['accuracy'], label='Training Accuracy')
plt.plot(nn_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# %%
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 2)
plt.plot(nn_history.history['loss'], label='Training Loss')
plt.plot(nn_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# %%
nn_clfr.evaluate(X_test,y_test)

# %% [markdown]
# ### **Importing sklearn for the ensemble models**

# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import seaborn as sns

# %% [markdown]
# Using a bagging ensemble technique by training a random forest classifier with 50 decision trees

# %%
rf_clfr = RandomForestClassifier(n_estimators=50)
rf_clfr.fit(X_train,y_train)

# %% [markdown]
# Testing the trained random forest classifier

# %%
rf_clfr.score(X_test,y_test)

# %%
from sklearn.tree import export_graphviz
from graphviz import Source

# Select a single tree from the Random Forest
estimator = rf_clfr.estimators_[0]  # Select the first tree

# Export the selected tree to a .dot file
export_graphviz(
    estimator,
    out_file='../Results/random_forest_tree.dot',
    feature_names=['MQ-5', 'MQ-135', 'MQ-2', 'MQ-3', 'TD'],  # Replace with your feature names
    class_names=sorted(data['SCENARIO'].unique()),  # Replace with your class names
    rounded=True,
    proportion=False,
    precision=2,
    filled=True
)

# Load the .dot file and convert it to PNG
dot_data = open('../Results/random_forest_tree.dot').read()
graph = Source(dot_data)
graph.render('random_forest_tree', format='png', cleanup=True)

# %%
y_true= y_test
y_pred= rf_clfr.predict(X_test)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.cividis)
plt.show()

# %%
precision = precision_score(y_test, y_pred, average='macro')  # 'macro' for multi-class
recall = recall_score(y_test, y_pred, average='macro')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# %%
y_true= y_test
y_pred= knn_clfr.predict(X_test)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.cividis)
plt.show()

# %%
precision = precision_score(y_test, y_pred, average='macro')  # 'macro' for multi-class
recall = recall_score(y_test, y_pred, average='macro')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# %% [markdown]
# Using a boosting ensemble technique by training a gradient boosting classifier and extreme gradient boosting classifier

# %%
gb_clfr = GradientBoostingClassifier()
gb_clfr.fit(X_train,y_train)

xgb_clfr = XGBClassifier()
xgb_clfr.fit(X_train,y_train)

# %% [markdown]
# Testing the trained gradient boosting classifier

# %%
gb_clfr.score(X_test,y_test)

# %%
y_true= y_test
y_pred= gb_clfr.predict(X_test)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.cividis)
plt.show()

# %% [markdown]
# Testing the trained xgboost classifer

# %%
xgb_clfr.score(X_test,y_test)

# %% [markdown]
# Using a stacking ensemble technique that trains multiple base models:\
# --- a decision tree classifier and\
# --- a k-neighbors classifier\
# and then trains a logistic regression classifier to classify\
# based on the predictions of the base models

# %%
stack_clfr = StackingClassifier(estimators=[
    ('decision_tree',DecisionTreeClassifier()),
    ('k_neighbors',KNeighborsClassifier()),
],final_estimator=LogisticRegression(max_iter=100),n_jobs=3)

stack_clfr.fit(X_train,y_train)

# %% [markdown]
# Testing the trained stacking classifier 

# %%
stack_clfr.score(X_test,y_test)

# %%
y_true= y_test
y_pred= stack_clfr.predict(X_test)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.cividis)
plt.show()

# %% [markdown]
# ### **Visualizing the decision tree model**

# %%
from sklearn import tree

# %%
tree.export_graphviz(tree_clfr,
                     out_file='../Results/decision_tree_viz.dot',
                     feature_names=['MQ-5','MQ-135','MQ-2','MQ-3','TD'],
                     class_names=sorted(data['SCENARIO'].unique()),
                     label='all',
                     rounded=True,
                     filled=True
                     )


# %% [markdown]
# ### **Saving all the models**

# %%
import joblib

# %%
joblib.dump(log_reg,'../Trained_Models/logistic_regression.pkl')
joblib.dump(tree_clfr,'../Trained_Models/decision_tree.pkl')
joblib.dump(knn_clfr,'../Trained_Models/k_neighbors.pkl')
joblib.dump(rf_clfr,'../Trained_Models/random_forest.pkl')
joblib.dump(gb_clfr,'../Trained_Models/gradient_boost.pkl')
joblib.dump(xgb_clfr,'../Trained_Models/x_gradient_boost.pkl')
joblib.dump(stack_clfr,'../Trained_Models/stacking.pkl')
nn_clfr.save('../Trained_Models/neural_network.keras')

# %% [markdown]
# ### **Accuracy Metrics**
# Logistic Regression : 0.5\
# Decision Tree : 0.9807692307692307\
# KNeighbors : 0.9615384615384616\
# Random Forest : 0.9935897435897436\
# Gradient Boosting : 0.9871794871794872\
# Extreme Gradient Boosting : 1.0\
# Stacking : 0.9871794871794872\
# Neural Network : 0.9807692170143127


