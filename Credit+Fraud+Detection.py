
# coding: utf-8

# # CREDIT CARD FRAUD DETECTION

# This following notebook will help us analyze the Credit Card Fraud Detection Classes and the following models will be used to test the accuracy of fraudulent transactions.
# 
# 1. Random Forest Classifier
# 2. Decision Tree Classifier (CART)
# 3. XG Boost Algorithm

# In[26]:

#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display #import display for DataFrame usage
from sklearn.metrics import confusion_matrix
import itertools
import collections
from sklearn.preprocessing import normalize
from sklearn import tree
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from subprocess import check_output

get_ipython().magic('matplotlib inline')


# # Data Input

# In[27]:

data = pd.read_csv("E:/School/Sem 2/Knowledge Discovery in Databases/Final Project/Work/creditcard.csv")
data.head()


# # Assesment of the Target Class

# In[28]:

count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# * First Pass: Random Forest with all columns

# In[29]:

data_class_outcomes = data['Class']
#preserving only necessary columns 
data.drop(['Class'], axis = 1, inplace = True)


# In[30]:

#import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,data_class_outcomes,test_size=0.25, random_state=42)
print("Training and testing split was successful.")


# In[31]:

#Classifier = RFC
def implement_rfc(X_train,y_train,X_test):
    """
    This function fits and transforms data using 
    Random Forest Classifier technique and 
    returns the y_pred value
    """
    clf_B = RandomForestClassifier(n_estimators=98)
    clf_B.fit(X_train, y_train)
    y_pred = clf_B.predict(X_test)
    return y_pred

y_pred = implement_rfc(X_train,y_train,X_test)


# In[32]:

def calculate_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)

result_confusion_matrix = calculate_confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
class_names = [0,1]
plot_confusion_matrix(result_confusion_matrix, classes=class_names,title='Confusion matrix, with all features, <time> and <amount>')


# In[33]:

def calculate_add_scores(confusion_matrix,Classifier="RFC"):
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    precision = (TP/TP+FP)
    recall = (TP/TP+FN)
    values = [{'Classifier':Classifier,'Accuracy':accuracy,'Precision':precision,
              'Recall':recall}]
    dataframe = pd.DataFrame(values,columns=values[0].keys())
    return dataframe

df = calculate_add_scores(result_confusion_matrix)
print(df)


# * Second Pass: Random Forest on dropping column 'TIME'

# In[34]:

data_time_outcomes = data['Time']
#preserving only necessary columns, dropping 'Time' 
data.drop(['Time'], axis = 1, inplace = True)


# In[35]:

data.describe()


# In[36]:

#import train_test split 
X_train, X_test, y_train, y_test = train_test_split(data,data_class_outcomes,test_size=0.25, random_state=42)
print("Training and testing split was successful.")


# In[37]:

y_pred = implement_rfc(X_train,y_train,X_test)


# In[38]:

confusion_matrix_1 = calculate_confusion_matrix(y_test,y_pred)
class_names = [0,1]
plot_confusion_matrix(confusion_matrix_1, normalize=False, classes=class_names,
                      title='Confusion matrix, with all dimensions except <time> ')


# In[39]:

df1 = calculate_add_scores(confusion_matrix_1)
frames = [df,df1]
df = pd.concat(frames)
print(df)


# * Pass 3: Random Forest on dropping both 'Time' & 'Amount', preserving only features

# In[40]:

data_amount_outcomes = data['Amount']
data.drop(['Amount'], axis = 1, inplace = True)


# In[41]:

display(data.describe())


# In[42]:

#import train_test split 
X_train, X_test, y_train, y_test = train_test_split(data,data_class_outcomes,test_size=0.25, random_state=42)
print("Training and testing split was successful.")


# In[43]:

y_pred = implement_rfc(X_train,y_train,X_test)
confusion_matrix_2 = calculate_confusion_matrix(y_test,y_pred)
class_names = [0,1]
plot_confusion_matrix(confusion_matrix_2, normalize=False, classes=class_names,
                      title='Confusion matrix, with only features, no <time> and no <Amount> ')


# In[44]:

df2 = calculate_add_scores(confusion_matrix_2)
frames = [df,df2]
df = pd.concat(frames)
print(df)


# * Now the data is normalized to check accuracy after Data Handling

# In[45]:

normalize_array = normalize(data_amount_outcomes.values.reshape(1,-1))


# * Pass 4: Random Forest with all features, no 'Time' and Normalized 'Amount'

# In[46]:

#Concatenate data using Numpy
new_data = np.concatenate((data, normalize_array.T), axis=1)


# * Training set = 75%, Test set = 25%

# In[47]:

#import train_test split 
X_train, X_test, y_train, y_test = train_test_split(data,data_class_outcomes,test_size=0.25, random_state=42)
print("Training and testing split was successful.")


# In[48]:

clf = RandomForestClassifier(n_estimators=98)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix_3 = calculate_confusion_matrix(y_test,y_pred)
class_names = [0,1]
plot_confusion_matrix(confusion_matrix_3, normalize=False, classes=class_names,
                      title='Confusion matrix, with all dimensions but no <time>, includes normalized <Amount> ')


# In[49]:

df3 = calculate_add_scores(confusion_matrix_3)
frames = [df,df3]
df = pd.concat(frames)
print(df)


# * Pass 5: Random Forest with all features, no 'Time' and Normalized 'Amount' 
# * Training set = 80%, Test set = 20%

# In[96]:

#try 2 with different parameters
X_train, X_test, y_train, y_test = train_test_split(new_data,data_class_outcomes,test_size=0.2, random_state=42)
print("Training and testing split was successful.")
clf = RandomForestClassifier(n_estimators=98)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix_4 = calculate_confusion_matrix(y_test,y_pred)
class_names = [0,1]
plot_confusion_matrix(confusion_matrix_4, normalize=False, classes=class_names,
                      title='Confusion matrix, with all dimensions but no <time>, includes normalized <Amount> ')


# In[95]:

importance = clf.feature_importances_
print(importance)


# In[51]:

df4 = calculate_add_scores(confusion_matrix_4)
frames = [df,df4]
df = pd.concat(frames)
print(df)


# * Decision Tree Classifier with Max_Depth = 6

# In[52]:

X_train, X_test, y_train, y_test = train_test_split(new_data,data_class_outcomes,test_size=0.2, random_state=42)
print("Training and testing split was successful.")
clf = tree.DecisionTreeClassifier(random_state=42,max_depth=6)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix_4 = calculate_confusion_matrix(y_test,y_pred)
class_names = [0,1]
plot_confusion_matrix(confusion_matrix_4, normalize=False, classes=class_names,
                      title='Confusion matrix, with all dimensions but no <time> and normalized <Amount> ')


# In[53]:

df5 = calculate_add_scores(confusion_matrix_4,Classifier="DTC-1")
frames = [df,df5]
df = pd.concat(frames)
print(df)


# * Decision Tree Classifier with Max Depth = 7

# In[54]:

X_train, X_test, y_train, y_test = train_test_split(new_data,data_class_outcomes,test_size=0.2, random_state=42)
print("Training and testing split was successful.")
clf = tree.DecisionTreeClassifier(random_state=42,max_depth=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix_5 = calculate_confusion_matrix(y_test,y_pred)
class_names = [0,1]
plot_confusion_matrix(confusion_matrix_5, normalize=False, classes=class_names,
                      title='Confusion matrix, with all dimensions but <time> and normalized <Amount> ')


# In[87]:

dotfile = open("D:/clf.dot", 'w')
tree.export_graphviz(clf, out_file = dotfile, feature_names = data.columns,class_names = ['legit','fraud'])
dotfile.close()


# In[84]:

from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
feature_names=data.columns,
filled=True, rounded=True,
special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
print(graph)


# In[55]:

df6 = calculate_add_scores(confusion_matrix_5,Classifier="DTC-2")
frames = [df,df6]
df = pd.concat(frames)
print(df)


# # Analysis

# Above result explanation with dimesnions 
# 
# * RFC = Random Forest Classifier
# 
# * DTC = Decision Tree Classifier
# 
# 1) First Pass - RFC  including all dimensions in data set with test_size =0.25
# 
# 2) Second Pass - RFC including all dimensions but time in data set with test_size =0.25
# 
# 3) Third Pass - RFC including all dimensions but (time,amount) in data set with test_size =0.25
# 
# 4) Fourth Pass - RFC including all dimensions but time, includes normalized amount with test_size =0.25
# 
# 5) Fifth Pass - RFC including all dimensions but time, includes normalized amount with test_size =0.2
# 
# 6) Sixth Pass - DTC with max_depth=6, including all dimensions but time, includes normalized amount with test_size =0.2               
# 
# 7) Seventh Pass - DTC with max_depth=7, including all dimensions but time, includes normalized  amount with test_size =0.2
#                

# # The best accuracy is obtained in the Random Forest Classifier

# * The optimal results are obtained in the fifth and sixth pass due to Precision and Recall

# # XGBoost (3rd classifier)

# Data Input

# In[15]:

dataset = pd.read_csv("E:/School/Sem 2/Knowledge Discovery in Databases/Final Project/Work/creditcard.csv")
dataset.head()


# In[16]:

dataset.describe()


# In[17]:

print(len(dataset[dataset.Class == 1]))
features = dataset.iloc[:, :-1]
print(features.shape)
label = dataset.iloc[:, -1].values
print(label.shape)

# heatmap for correlation, verifying that pca is already done
corrMat = features.corr()
sns.heatmap(corrMat, vmax=0.8)


# Feature Engineering & Scaling

# In[18]:

fraudInd = np.asarray(np.where(label == 1))
noFraudInd = np.where(label == 0)
features = features.values

# data standarization (zero-mean, unit variance) ~ truncation to [-1, 1]
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)


# In[20]:

TestPortion = 0.2
RND_STATE = 1

x_tr, x_test, y_tr, y_test = train_test_split(features, label, test_size = TestPortion, random_state = 1)

xgb_model = xgb.XGBClassifier(n_estimators=100)
xgb_model.fit(x_tr, y_tr, verbose = 1)

y_pred = xgb_model.predict(x_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
area = auc(recall, precision)

print('------------ Results for XGBClassifier ---------------')
print('cm:', confusion_matrix(y_test,y_pred))
#print('cr:', classification_report(y_test,y_pred))
#print('recall_score:', recall_score(y_test,y_pred))
print('roc_auc_score:',roc_auc_score(y_test,y_pred))
print("Area Under P-R Curve: ",area)

