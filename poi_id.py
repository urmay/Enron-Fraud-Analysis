#!/usr/bin/python
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import warnings
import random
import pprint 

#==============================================================================
# #loading the features
#==============================================================================
warnings.filterwarnings('ignore')

financial_features = ['salary', 'deferral_payments', 'total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred',
                      'deferred_income','total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']

features_list = poi_label + financial_features + email_features
pprint.pprint(features_list)
print(len(features_list))
#20
#this are the total no of features we have.
#==============================================================================
# ### Load the dictionary containing the dataset
#==============================================================================
from sklearn.externals import joblib

data_dict = joblib.load("final_project_dataset.pkl")

#Another way to load the dictionary
### Load the dictionary containing the dataset
#with open("final_project_dataset.pkl", "rb") as data_file:
#    data_dict = pickle.load(data_file)
#data_dict_df = pd.DataFrame(data_dict)


#==============================================================================
# #converting data_dict into dataframe format to use 
# #few function which are available only with pandas inorder to explore the dataset.
#==============================================================================

data_dict_df=pd.DataFrame(data_dict)
#Checking what are features available in the datset and how many records were there. 
data_dict_df.head(25)

#==============================================================================
# # How many poi are there in the dataset who were having POI = 1.
#==============================================================================

total_no_of_people=len(data_dict)
print("Total No of People in the dataset:",total_no_of_people)


poi =0
for person in data_dict:
    if data_dict[person]['poi'] == True:
        poi +=1

print("Total No of poi:{0}".format(poi))

#now total no of none poi 
print("Total No of Non poi:{0}".format(total_no_of_people-poi))

#Total no of features in the dataset


total_features = data_dict[data_dict.keys()[0]].keys()
print("\nThere are total {0} features for every individual of the datasetout of which {1} features are used,one is target feature"
      .format(len(total_features),len(features_list)))

#==============================================================================
# # Are there any features with missing values?
#==============================================================================

missing_values = {}
for feature in total_features:
    missing_values[feature] = 0
    
for person in data_dict:
   for feature in data_dict[person]:
       if data_dict[person][feature] == "NaN":
           missing_values[feature] += 1
           
total_missing_val=[]
for feature in sorted(missing_values):
    value=missing_values[feature]

    print("{0}:{1}".format(feature,value))
    total_missing_val.append(value)
    
total=sum(total_missing_val)
print("Total No of Missing Values:",total)


#==============================================================================
#                      ### Task 2: Remove outliers
#==============================================================================

import matplotlib.pyplot as plt
def plot():
    feature_select =["bonus","salary"]
    data = featureFormat(data_dict,feature_select)
    import matplotlib.pyplot as plt
    for point in data:
        salary = point[0]
        bonus = point[1]
        plt.scatter(salary,bonus)
    
    plt.xlabel("Salary")
    plt.ylabel("Bonus")
#    plt.savefig("plt2.png")
    plt.show()
    
#plot()

#Clear outlier is visible from the above plot

outliers_salary = []
for key in data_dict:
    val = data_dict[key]["salary"]
    if val == "NaN":
        continue
    outliers_salary.append((key,int(val)))
    
print(sorted(outliers_salary,key = lambda x:x[1],reverse = True)[:5])

outliers_bonus = []
for key in data_dict:
    val = data_dict[key]["bonus"]
    if val == "NaN":
        continue
    outliers_bonus.append((key,int(val)))

print(sorted(outliers_bonus,key = lambda x:x[1],reverse = True)[:5])
# see the key and their value for total,only total is making huge difference so removing it.

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

#removing it
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
print(len(data_dict))
#now ploting after outlier removal
#we have already made a function so calling plot() function again

#plot()

def plot_2():
    feature_select_2 =["from_this_person_to_poi", "from_poi_to_this_person"]
    data = featureFormat(data_dict,feature_select_2)

    for point in data:
        from_this_person_to_poi = point[0]
        from_poi_to_this_person = point[1]
        plt.scatter(from_this_person_to_poi,from_poi_to_this_person)
    
    plt.xlabel("from_this_person_to_poi")
    plt.ylabel("from_poi_to_this_person")
    plt.savefig("plt3.png")
    plt.show()
    
#plot_2()

#from the graph you can see four outliers but after investigating it is found that they are real person.
#so we will not remove it from the dataset.

data_dict_df=pd.DataFrame(data_dict)
data_dict_df.shape

#==============================================================================
# #Dealing with Nan
#==============================================================================

#converting NaN values to 0
df_values = pd.DataFrame.from_records(list(data_dict.values()))
df_values.head()

df_null=df_values.replace(to_replace = "NaN",value = 0,inplace =True)
df_null =df_values.fillna(0).copy(deep =True)
df_null.columns = list(df_values.columns.values)
#print df_null.isnull().sum()
print df_null.head()

df_null.describe()
 
### Now, since all the outliers have been removed, proceeding with next task.

#==============================================================================
# ###  checking the Accuracy,Precision,Recall before adding new feature
#==============================================================================
my_dataset = data_dict
len(my_dataset)
### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

random.seed(46)

scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
# as the datasset is skwed .so there are less no of poi =1 
# we have tried chnging the test_size =0.3  and 0.4 for the experimenatal purpose.

#features_train,features_test,labels_train,labels_test =cross_validation.train_test_split(features,labels,test_size =0.3,random_state =42)
#clf = DecisionTreeClassifier(max_depth=3)
#It gives best accuracy when  max_depth = 3
#clf.fit(features_train,labels_train)


#pred = clf.predict(features_test)
#accuracy = accuracy_score(labels_test,pred)
#print("Accuracy:",accuracy)
#('Accuracy:', 0.86046511627906974) for test_size =0.3
#('Accuracy:', 0.86206896551724133) for test_size = 0.4
#print("Precision :",precision_score(labels_test,pred))
#('Precision :', 0.33333333333333331)
#('Precision :', 0.33333333333333331)
#print("Recall :",recall_score(labels_test,pred))
#('Recall :', 0.20000000000000001)
#('Recall :', 0.14285714285714285)


# here in this case for the experimental purpose we took decision tree classifier 
#just to cheak before adding new feature what is the value of accuracy,precision and recall.

#this excercise it just to examine the performance 
#above result does not match our result requirement in the case of Recall

### Store to my_dataset for easy export below.
#now we will  add new feature

#==============================================================================
# # ### Task 3: Create new feature(s)   
#==============================================================================

def add_savings(my_dataset,features_list):
    fields=['salary','expenses']
    for record in data_dict:
        person = data_dict[record]
        is_valid =True
        for field in fields:
            if person[field]=="NaN":
                is_valid = False
            
            if is_valid:
                savings = person['salary'] -\
                          float(person['expenses'])
                
                person['savings']=savings
            else:
                person['savings']="NaN"
    features_list +=['savings']



   
def add_poi_ratio(data_dict, features_list):
    """ mutates data dict to add proportion of email interaction with pois """
    fields = ['to_messages', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages'] +\
                             person['from_messages']
            poi_messages = person['from_poi_to_this_person'] +\
                           person['from_this_person_to_poi']
            person['poi_ratio'] = float(poi_messages) / total_messages
        else:
            person['poi_ratio'] = 'NaN'
    features_list += ['poi_ratio']



def add_fraction_to_poi(data_dict, features_list):
    """ mutates data dict to add proportion of email fraction_to_poi """
    fields = ['from_messages', 'from_this_person_to_poi']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['from_messages']
            poi_messages =   person['from_this_person_to_poi']
            person['fraction_to_poi'] = float(poi_messages) / total_messages
        else:
            person['fraction_to_poi'] = 'NaN'
    features_list += ['fraction_to_poi']


def add_fraction_from_poi(data_dict, features_list):
    """ mutates data dict to add proportion of email fraction_from_poi """
    fields = ['to_messages', 'from_poi_to_this_person']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages']
            poi_messages =   person['from_poi_to_this_person']
            person['fraction_from_poi'] = float(poi_messages) / total_messages
        else:
            person['fraction_from_poi'] = 'NaN'
    features_list += ['fraction_from_poi']



#Adding them to the features list
add_poi_ratio(data_dict, features_list)
add_fraction_to_poi(data_dict, features_list)
add_fraction_from_poi(data_dict, features_list)
add_savings(data_dict,features_list)
pprint.pprint(features_list)
print(len(features_list))    

#now we have 24 feature list in our dataset.
#what we need to do next is to identify the k best features for the best results
#==============================================================================
#when using all 24 features
#Accuracy : 0.86046511627906974
#Precision : 0.40000000000000002
#Recall : 0.40000000000000002

 
data = featureFormat(my_dataset,features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
features_train,features_test,labels_train,labels_test =cross_validation.train_test_split(features,labels,test_size =0.3,random_state =42)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(features_train,labels_train)


pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test,pred)
print("Accuracy:",accuracy)
#('Accuracy:', 0.86046511627906974)
# 
print("Precision :",precision_score(labels_test,pred))
# #('Precision :',  0.40000000000000002))
#     
print("Recall :",recall_score(labels_test,pred))
# #('Recall :', 0.40000000000000002)

#but we want  k best features

#==============================================================================
#now we will use selectkbest algorithm to select k best features inorder to find best prediction results.
def takeSecond(elem):
    """ take second element for sort
    """
    return elem[1]


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# intelligently select features (univariate feature selection)
from sklearn.feature_selection import f_classif
selector = SelectKBest(f_classif, k = 10)
selector.fit(features, labels)
scores = zip(features_list[1:], selector.scores_)
sorted_scores = sorted(scores, key = takeSecond, reverse = True)
pprint.pprint(sorted_scores)
target_label ='poi'
kBest_features_list = [target_label] + [(i[0]) for i in sorted_scores[0:10]]
 
for person in data_dict:
    for value in data_dict[person]:
        if data_dict[person][value] == 'NaN':
            # fill NaN values
            data_dict[person][value] = 0

my_dataset = data_dict

# dataset with k best features
from sklearn import preprocessing
data = featureFormat(my_dataset, kBest_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# dataset with k best features with added 2 new features
kBest_new_features_list = kBest_features_list + ['fraction_from_poi', 'shared_receipt_with_poi']
data = featureFormat(my_dataset, kBest_new_features_list, sort_keys = True)
new_labels, new_features = targetFeatureSplit(data)
new_features = scaler.fit_transform(new_features)
#==============================================================================
# 
#==============================================================================

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
def tune_params(grid_search, features, labels, params, iters = 10):
    """ given a grid_search and parameters list (if exist) for a specific model,
    along with features and labels list,
    it tunes the algorithm using grid search and prints out the average evaluation metrics
    results (accuracy, percision, recall) after performing the tuning for iter times,
    and the best hyperparameters for the model
    """
    accuracy = []
    #p = []
    #r = []
    precision =[]
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size = 0.4, random_state = i)
        grid_search.fit(features_train, labels_train)
        predicts = grid_search.predict(features_test)

        accuracy = accuracy + [accuracy_score(labels_test, predicts)] 
        #p = precision_score(labels_test,predicts,average ='micro')
        #r = recall_score(labels_test,predicts,average = 'micro')
        precision = precision + [precision_score(labels_test, predicts)]
        recall = recall + [recall_score(labels_test, predicts)]
    print ("accuracy: {}".format(np.mean(accuracy)))
    print ("precision: {}".format(np.mean(precision)))
    print ("recall: {}".format(np.mean(recall)))
    #if want to see the confusion matrix
    cnf_matrix = confusion_matrix( labels_test,predicts)
    print("Confusion Matrix:\n{0}".format(cnf_matrix))
    
    best_params = grid_search.best_estimator_.get_params()
    print(best_params)
    for param_name in params.keys():
        print("{0} = {1}, " .format(param_name, best_params[param_name]))
        


#==============================================================================
# # 1. Decision Tree
#==============================================================================

#==============================================================================
# clf = DecisionTreeClassifier(max_depth = 2)
# clf.fit(features_train,labels_train)
# pred = clf.predict(features_test)
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(labels_test,pred)
# print("Decision Tree Classifier: ")
# print "Accuracy: " + str(accuracy)
# print "Precision Score: " + str(precision_score(labels_test,pred))
# print "Recall Score: " + str(recall_score(labels_test,pred))
# 
# Decision Tree Classifier: 
# Accuracy: 0.860465116279
# Precision Score: 0.4
# Recall Score: 0.4
#==============================================================================


from sklearn import tree
t0=time()
clf = tree.DecisionTreeClassifier()
dt_param = {'criterion':('gini', 'entropy'),
'splitter':('best','random'),
'max_depth':[2,3,4,5,6],
'min_samples_split':[2,3]}
#we have also given max_depth to limited value as we dont want to overfit the tree. 
#we have given min_samples_split : 2,3 as there are only few poi=1 in dataset so 
#using low min_samples_split we can classify.
dt_grid_search = GridSearchCV(estimator = clf, param_grid = dt_param)

print("Decision Tree model evaluation")
#tune_params(dt_grid_search, features, labels, dt_param)

tune_params(dt_grid_search,new_features,new_labels,dt_param)
print("Processing time:",round(time()-t0,3),"s")

#we can also count same for new parameters copying same and counting individually!

#==============================================================================
# # 2. Support Vector Machines
#==============================================================================

#==============================================================================
# clf = SVC(gamma=1, C=1)
# clf.fit(features_train,labels_train)
# pred = clf.predict(features_test)
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(labels_test,pred)
# print("SVM Classifier: ")
# print "Accuracy: " + str(accuracy)
# print "Precision Score: " + str(precision_score(labels_test,pred))
# print "Recall Score: " + str(recall_score(labels_test,pred))
# 
# SVM Classifier: 
# Accuracy: 0.883720930233
# Precision Score: 0.0
# Recall Score: 0.0
#==============================================================================

##from sklearn import svm
##t0=time()
##clf = svm.SVC()
##svm_param = {'kernel':('linear', 'rbf', 'sigmoid'),
##'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
##'C': [0.1, 1, 10, 100, 1000]}
##svm_grid_search = GridSearchCV(estimator = clf, param_grid = svm_param)
##print("SVM model evaluation")
###tune_params(svm_grid_search, features, labels, svm_param)
##
##tune_params(svm_grid_search,new_features,new_labels,svm_param)
##print("Processing time:",round(time()-t0,3),"s")

#==============================================================================
# # 3. Naive Bayes
#==============================================================================

#==============================================================================
# clf = GaussianNB()
# clf.fit(features_train,labels_train)
# pred = clf.predict(features_test)
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(labels_test,pred)
# print("Naive Bayes Classifier: ")
# print "Accuracy: " + str(accuracy)
# print "Precision Score: " + str(precision_score(labels_test,pred))
# print "Recall Score: " + str(recall_score(labels_test,pred))

# Naive Bayes Classifier: 
# Accuracy: 0.558139534884
# Precision Score: 0.15
# Recall Score: 0.6
#==============================================================================

##t0=time()
##clf = GaussianNB()
##nb_param = {}
##nb_grid_search = GridSearchCV(estimator = clf, param_grid = nb_param)
##
##print("Naive Bayes model evaluation")
###tune_params(nb_grid_search, features, labels, nb_param)
##tune_params(nb_grid_search,new_features,new_labels,nb_param)
##print("Processing time:",round(time()-t0,3),"s")

#==============================================================================
# # 4. Random Forest
#==============================================================================
##t0 =time()
##clf = RandomForestClassifier(n_estimators=10)
##rf_param = {'criterion':('gini','entropy'),
##            'max_depth':[3,4,5,6],
##            'min_samples_split':[2,3]
##            }
##rf_grid_search = GridSearchCV(estimator = clf, param_grid = rf_param)
##
##print("Random Forest model evaluation")
###tune_params(rf_grid_search, features, labels, rf_param)
##tune_params(rf_grid_search,new_features,new_labels,rf_param)
##print("Processing time:",round(time()-t0,3),"s")

#==============================================================================
# # 5. KNN
#==============================================================================
#==============================================================================
# If you want to find out the accuracy ,precision, and recall for individual k 

# for k in range(10):
#     k_value = k+1
#     neigh = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', algorithm='auto')
#     neigh.fit(features_train,labels_train) 
#     y_pred = neigh.predict(features_test)
#     print "Accuracy is ", accuracy_score(labels_test,y_pred)*100,"% for K-Value:",k_value
#     print "Precision is",precision_score(labels_test,y_pred),"% for k-value:",k_value
#     print "Recall is",recall_score(labels_test,y_pred),"% for k-value:",k_value
# 
# clf = KNeighborsClassifier(1)
# clf.fit(features_train,labels_train)
# pred = clf.predict(features_test)
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(labels_test,pred)
# print("kNN Classifier: ")
# print "Accuracy: " + str(accuracy)
# print "Precision Score: " + str(precision_score(labels_test,pred))
# print "Recall Score: " + str(recall_score(labels_test,pred))

# kNN Classifier: 
# Accuracy: 0.883720930233
# Precision Score: 0.5
# Recall Score: 0.2
#==============================================================================

##print("\nK Neighbors Execution and Evaluation\n")
##t0=time()
##k=np.arange(10)+1
##clf = KNeighborsClassifier()
##knn_param ={'n_neighbors':k}
##knn_grid_search  = GridSearchCV(estimator =clf,param_grid =knn_param)
##print("KNN evaluation")
###tune_params(knn_grid_search, features, labels, knn_param)
##
##tune_params(knn_grid_search,new_features,new_labels,knn_param)
##print("Processing time:",round(time()-t0,3),"s")

#==============================================================================
# # 6. AdaBoost
#==============================================================================

#==============================================================================
# clf = AdaBoostClassifier()
# clf.fit(features_train,labels_train)
# pred = clf.predict(features_test)
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(labels_test,pred)
# print("Adaboost Classifier: ")
# print "Accuracy: " + str(accuracy)
# print "Precision Score: " + str(precision_score(labels_test,pred))
# print "Recall Score: " + str(recall_score(labels_test,pred))
# 
# Adaboost Classifier: 
# Accuracy: 0.837209302326
# Precision Score: 0.25
# Recall Score: 0.2
#==============================================================================

##print("\nAdaBoost Execution and Evaluation\n")
##t0=time()
##clf = AdaBoostClassifier(learning_rate =1.5,n_estimators = 9, algorithm ='SAMME.R')
##ada_param = {}
##ada_grid_search = GridSearchCV(estimator = clf,param_grid =ada_param)
###tune_params(ada_grid_search, features, labels, ada_param)
##
##tune_params(ada_grid_search,new_features,new_labels,ada_param)
##print("Processing time:",round(time()-t0,3),"s")


#==============================================================================


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

features_list = kBest_features_list
dump_classifier_and_data(clf, my_dataset, features_list)
