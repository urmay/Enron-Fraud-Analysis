
# Identify Fraud from Enron Email

## Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity. [[1](https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475461/lessons/3174288624239847/concepts/31803986370923)]

## 1. Understanding the Dataset and Question

> Q1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

### 1.1 Data Exploration

Goal of the project is to develop a POI(person of interest) identifier based enron data set containing financial and email data that was made public as a result of the Enron Bankrupcy Scandal.The enron data set contains email and financial data of 146 executives which will be used to identify POI. "Person of interest" is a term used by U.S. law enforcement when identifying someone involved in a criminal investigation who has not been arrested or formally accused of a crime. [[1](https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475461/lessons/3174288624239847/concepts/31803986370923) ]
In this report different machine learning techniques are used to build the POI identifier.

The dataset contains 146 user data with 14 financial features ,6 email features, 1 labeled feature(POI).
From the 146 records 18 were apriori labeled as POI (Person Of Interest).

** Total No of Data Points: **
<br>
There are total (146 * 21) data points

** Allocation across classes (POI/non-POI): **
<br>Out of 146 , 18 of them are POIs, while 128(146-18) are non-POIs

** Feature List: **
<br>In this project we have used financial features and email features for the analysis.

** Number of Features Used: **
<br>Per user/ person, there are 20 features available out of which 19 are financial feature and email feature and 1 is a binary categorization feature indicating whether a person is POI or not.

** Are there any missing values in the features ? How many for individual feature? **
<br>Yes there are many missing values or "NaN" in the project.There are total 1358 missing values.Below is the list of missing value in individual

bonus: 64<br>
deferral_payments: 107<br>
deferred_income: 97<br>
director_fees: 129<br>
email_address: 35<br>
exercised_stock_options: 44<br>
expenses: 51<br>
from_messages: 60<br>
from_poi_to_this_person: 60<br>
from_this_person_to_poi: 60<br>
loan_advances: 142<br>
long_term_incentive: 80<br>
other: 53<br>
poi: 0<br>
restricted_stock: 36<br>
restricted_stock_deferred: 128<br>
salary: 51<br>
shared_receipt_with_poi: 60<br>
to_messages: 60<br>
total_payments: 21<br>
total_stock_value: 20<br>

**Total number of missing values are :1358**<br>



```python
#Features

financial_features = ['salary', 'deferral_payments', 'total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred',
                      'deferred_income','total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']

features_list = poi_label + financial_features + email_features
print(features_list)

### Load the dictionary containing the dataset
from sklearn.externals import joblib
import pandas as pd

data_dict = joblib.load("final_project_dataset.pkl")

#converting data_file into dataframe format to use 
#few function which are available only with pandas inorder to explore the dataset.

data_dict_df=pd.DataFrame(data_dict)
data_dict_df.head(22)
#Checking what are features available in the datset and how many records were there. 

```

    ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ALLEN PHILLIP K</th>
      <th>BADUM JAMES P</th>
      <th>BANNANTINE JAMES M</th>
      <th>BAXTER JOHN C</th>
      <th>BAY FRANKLIN R</th>
      <th>BAZELIDES PHILIP J</th>
      <th>BECK SALLY W</th>
      <th>BELDEN TIMOTHY N</th>
      <th>BELFER ROBERT</th>
      <th>BERBERIAN DAVID</th>
      <th>...</th>
      <th>WASAFF GEORGE</th>
      <th>WESTFAHL RICHARD K</th>
      <th>WHALEY DAVID A</th>
      <th>WHALLEY LAWRENCE G</th>
      <th>WHITE JR THOMAS E</th>
      <th>WINOKUR JR. HERBERT S</th>
      <th>WODRASKA JOHN</th>
      <th>WROBEL BRUCE</th>
      <th>YEAGER F SCOTT</th>
      <th>YEAP SOON</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bonus</th>
      <td>4175000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1200000</td>
      <td>400000</td>
      <td>NaN</td>
      <td>700000</td>
      <td>5249999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>325000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3000000</td>
      <td>450000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>deferral_payments</th>
      <td>2869717</td>
      <td>178980</td>
      <td>NaN</td>
      <td>1295738</td>
      <td>260455</td>
      <td>684694</td>
      <td>NaN</td>
      <td>2144013</td>
      <td>-102500</td>
      <td>NaN</td>
      <td>...</td>
      <td>831299</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>deferred_income</th>
      <td>-3081055</td>
      <td>NaN</td>
      <td>-5104</td>
      <td>-1386055</td>
      <td>-201641</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2334434</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>-583325</td>
      <td>-10800</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-25000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>director_fees</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3285</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>108579</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>email_address</th>
      <td>phillip.allen@enron.com</td>
      <td>NaN</td>
      <td>james.bannantine@enron.com</td>
      <td>NaN</td>
      <td>frank.bay@enron.com</td>
      <td>NaN</td>
      <td>sally.beck@enron.com</td>
      <td>tim.belden@enron.com</td>
      <td>NaN</td>
      <td>david.berberian@enron.com</td>
      <td>...</td>
      <td>george.wasaff@enron.com</td>
      <td>dick.westfahl@enron.com</td>
      <td>NaN</td>
      <td>greg.whalley@enron.com</td>
      <td>thomas.white@enron.com</td>
      <td>NaN</td>
      <td>john.wodraska@enron.com</td>
      <td>NaN</td>
      <td>scott.yeager@enron.com</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>exercised_stock_options</th>
      <td>1729541</td>
      <td>257817</td>
      <td>4046157</td>
      <td>6680544</td>
      <td>NaN</td>
      <td>1599641</td>
      <td>NaN</td>
      <td>953136</td>
      <td>3285</td>
      <td>1624396</td>
      <td>...</td>
      <td>1668260</td>
      <td>NaN</td>
      <td>98718</td>
      <td>3282960</td>
      <td>1297049</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>139130</td>
      <td>8308552</td>
      <td>192758</td>
    </tr>
    <tr>
      <th>expenses</th>
      <td>13868</td>
      <td>3486</td>
      <td>56301</td>
      <td>11200</td>
      <td>129142</td>
      <td>NaN</td>
      <td>37172</td>
      <td>17355</td>
      <td>NaN</td>
      <td>11892</td>
      <td>...</td>
      <td>NaN</td>
      <td>51870</td>
      <td>NaN</td>
      <td>57838</td>
      <td>81353</td>
      <td>1413</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>53947</td>
      <td>55097</td>
    </tr>
    <tr>
      <th>from_messages</th>
      <td>2195</td>
      <td>NaN</td>
      <td>29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4343</td>
      <td>484</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>556</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>from_poi_to_this_person</th>
      <td>47</td>
      <td>NaN</td>
      <td>39</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>144</td>
      <td>228</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>186</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>from_this_person_to_poi</th>
      <td>65</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>386</td>
      <td>108</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>loan_advances</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>long_term_incentive</th>
      <td>304805</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1586055</td>
      <td>NaN</td>
      <td>93750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>200000</td>
      <td>256191</td>
      <td>NaN</td>
      <td>808346</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>other</th>
      <td>152</td>
      <td>NaN</td>
      <td>864523</td>
      <td>2660303</td>
      <td>69</td>
      <td>874</td>
      <td>566</td>
      <td>210698</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1425</td>
      <td>401130</td>
      <td>NaN</td>
      <td>301026</td>
      <td>1085463</td>
      <td>NaN</td>
      <td>189583</td>
      <td>NaN</td>
      <td>147950</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>poi</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>restricted_stock</th>
      <td>126027</td>
      <td>NaN</td>
      <td>1757552</td>
      <td>3942714</td>
      <td>145796</td>
      <td>NaN</td>
      <td>126027</td>
      <td>157569</td>
      <td>NaN</td>
      <td>869220</td>
      <td>...</td>
      <td>388167</td>
      <td>384930</td>
      <td>NaN</td>
      <td>2796177</td>
      <td>13847074</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3576206</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>restricted_stock_deferred</th>
      <td>-126027</td>
      <td>NaN</td>
      <td>-560222</td>
      <td>NaN</td>
      <td>-82782</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44093</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>salary</th>
      <td>201955</td>
      <td>NaN</td>
      <td>477</td>
      <td>267102</td>
      <td>239671</td>
      <td>80818</td>
      <td>231330</td>
      <td>213999</td>
      <td>NaN</td>
      <td>216582</td>
      <td>...</td>
      <td>259996</td>
      <td>63744</td>
      <td>NaN</td>
      <td>510364</td>
      <td>317543</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>158403</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>shared_receipt_with_poi</th>
      <td>1407</td>
      <td>NaN</td>
      <td>465</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2639</td>
      <td>5521</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>337</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3920</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>to_messages</th>
      <td>2902</td>
      <td>NaN</td>
      <td>566</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7315</td>
      <td>7991</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>400</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6019</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>total_payments</th>
      <td>4484442</td>
      <td>182466</td>
      <td>916197</td>
      <td>5634343</td>
      <td>827696</td>
      <td>860136</td>
      <td>969068</td>
      <td>5501630</td>
      <td>102500</td>
      <td>228474</td>
      <td>...</td>
      <td>1034395</td>
      <td>762135</td>
      <td>NaN</td>
      <td>4677574</td>
      <td>1934359</td>
      <td>84992</td>
      <td>189583</td>
      <td>NaN</td>
      <td>360300</td>
      <td>55097</td>
    </tr>
    <tr>
      <th>total_stock_value</th>
      <td>1729541</td>
      <td>257817</td>
      <td>5243487</td>
      <td>10623258</td>
      <td>63014</td>
      <td>1599641</td>
      <td>126027</td>
      <td>1110705</td>
      <td>-44093</td>
      <td>2493616</td>
      <td>...</td>
      <td>2056427</td>
      <td>384930</td>
      <td>98718</td>
      <td>6079137</td>
      <td>15144123</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>139130</td>
      <td>11884758</td>
      <td>192758</td>
    </tr>
  </tbody>
</table>
<p>21 rows × 146 columns</p>
</div>



### 1.2 Outlier Investigation
 Below is the plot before removing outlier.From the graph you can see the clear one point which stands out different compared to the other.For further analysis we have printed top five records.<br>
 [('TOTAL', 97343619), ('LAVORATO JOHN J', 8000000), ('LAY KENNETH L', 7000000), ('SKILLING JEFFREY K', 5600000), ('BELDEN TIMOTHY N', 5249999)]<br>We can find the clear difference from the above output. 
 <img src="plt1.png">
 <br>
 
 Below is the graph after emoving two things .One is TOTAL. Then counted which data point has most NaNs and sorted them and based on that, removed the  __"THE TRAVEL AGENCY IN THE PARK "__.
 <img src ="plt2.png">
 <br>
 Besides this we have tried ploting "from_this_person_to_poi" vs  "from_poi_to_this_person".Below is the found output.From the graph you can see four outliers but after investigating it is found that they are real person.so we will not remove it from the dataset.
 <img src = "plt3.png">

## 2. Optimize Feature Selection/Engineering
> Q2 .What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

### 2.1 Analyzing before adding new features


```python
### Now, since all the outliers have been removed, proceeding with next task.

#==============================================================================
# ###  checking the Accuracy,Precision,Recall before adding new feature
#==============================================================================
import random
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

random.seed(46)

scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# As the datasset is skewed having less no of poi =1 we have tried chnging the test_size =0.3 and 0.4 
#for the experimenatal purpose.

features_train,features_test,labels_train,labels_test =cross_validation.train_test_split(features,labels,test_size =0.3,random_state =42)
clf = DecisionTreeClassifier(max_depth=3)
#It gives best accuracy when  max_depth = 3

clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test,pred)
print("Accuracy:",accuracy)
#('Accuracy:', 0.86046511627906974) for test_size =0.3
#('Accuracy:', 0.86206896551724133) for test_size = 0.4
print("Precision :",precision_score(labels_test,pred))
#('Precision :', 0.33333333333333331)
#('Precision :', 0.33333333333333331)
print("Recall :",recall_score(labels_test,pred))
#('Recall :', 0.20000000000000001)
#('Recall :', 0.14285714285714285)

# here in this case for the experimental purpose we took decision tree classifier 
#just to cheak before adding new feature what is the value of accuracy,precision and recall.

#this excercise it just to examine the performance 
#above result does not match our result requirement in the case of Recall

### Store to my_dataset for easy export below.
#now we will  add new feature

```

    Accuracy: 0.860465116279
    Precision : 0.333333333333
    Recall : 0.2
    

### 2.2 Creating New Features

In this project we have engineered four new features.

    1. saving 
    2. poi_ratio
    3. fraction_from_poi
    4. fraction_to_poi
    
       - features['savings'] = float(salary-expenses).In words, saving of a person using simple subtraction between salary and expenses
       - features['fraction_to_poi'] = float(from_poi) / float(to_msg).In words, the ratio of the messages from POI to this person against all the messages sent to this person
       - features['fraction_from_poi'] = float(to_poi) / float(from_msg).In words, the ratio from this person to POI against all messages from this person.
       - feature['poi_ratio'] = float(poi_messages) / total_messages. In words,ratio of the total poi message to total no of messages.
    where,
    
 * total_messages = person['to_messages'] + person['from_messages']
 * poi_messages = person['from_poi_to_this_person'] + person['from_this_person_to_poi']
 * salary = features['salary']
 * expenses = features['expenses']

### 2.3 Intelligently select features

Now after adding we have total 24 features.but we want k best features which are most useful in this project.In order to decide which are the best features to use for POI Identifier, I utilized an feature selection function belonging to sklearn.feature_selection, i.e. SelectKBest, which automatically selects K features amongst other features which are most powerful, where K is a parameter such that if K=10, top 10 features will be selected.

I have done manual testing for choosing the value of k .I tried various values ranging from 6-10 but the value of K=10 suited the best as it gave the best results . After the selecion of K, I tried to get a near optimum value of number of features (again using manual testing and tuning) to be selected so that the algorithm performs at its best. I ended up choosing top 10 features. Below is the output of the function:

    



```python
 [
('exercised_stock_options', 24.815079733218194), 
('total_stock_value', 24.182898678566879),
('savings', 21.473791247049132),
('bonus', 20.792252047181535),
('salary', 18.289684043404513), 
('fraction_to_poi', 16.409712548035792), 
('deferred_income', 11.458476579280369),
('long_term_incentive', 9.9221860131898225), 
('restricted_stock', 9.2128106219771002), 
('total_payments', 8.7727777300916756), 
('shared_receipt_with_poi', 8.589420731682381), 
('loan_advances', 7.1840556582887247),
('expenses', 6.0941733106389453),
('poi_ratio', 5.399370288094401), 
('from_poi_to_this_person', 5.2434497133749582),
('other', 4.1874775069953749),
('fraction_from_poi', 3.1280917481567192),
('from_this_person_to_poi', 2.3826121082276739),
('director_fees', 2.1263278020077054), 
('to_messages', 1.6463411294420076),
('deferral_payments', 0.22461127473600989), 
('from_messages', 0.16970094762175533), 
('restricted_stock_deferred', 0.065499652909942141)
 ]
```


```python
#('Accuracy:', 0.82499999999999996)) for K=6
#('Accuracy:', 0.7857142857142857) for K=7
#(('Accuracy:', 0.86046511627906974) for K=8
#('Accuracy:', 0.7857142857142857) for K=9
#('Accuracy:', 0.84720930232558144) for K=10

#('Precision :', 0.25)k=6
#('Precision :', 0.20000000000000001)k=7
#('Precision :', 0.33333333333333331)k=8
#(('Precision :', 0.20000000000000001)k=9
#('Precision :', 0.33333333333333333)k=10

#('Recall :', 0.20000000000000001)k=6
#('Recall :', 0.16666666666666666)k=7
#('Recall :', 0.20000000000000001)k=8
#('Recall :', 0.16666666666666666)k=9
#('Recall :', 0.25)k=10
```


```python
# kbest_features_list with POI 

['poi', 'exercised_stock_options', 'total_stock_value', 'savings', 'bonus', 'salary', 'fraction_to_poi', 'deferred_income',
 'long_term_incentive','restricted_stock', 'total_payments']

#For the experimental purpose we have added two new features
# 1.fraction_from_poi (which we made)
# 2. shared_receipt_with_poi (which is having good impact of 8.58)

# kBest_new_features_list = kBest_features_list + ['fraction_from_poi', 'shared_receipt_with_poi']
['poi', 'exercised_stock_options', 'total_stock_value', 'savings', 'bonus', 'salary', 'fraction_to_poi', 'deferred_income', 
 'long_term_incentive', 'restricted_stock', 'total_payments', 'fraction_from_poi', 'shared_receipt_with_poi']

```

### 2.4 Feature scaling
<br>
Feature scaling is a method used to standardize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.[ [2](https://en.wikipedia.org/wiki/Feature_scaling)].

Since some of the features have extremely large values as compared to values of other selected features and the selected features have different units (USD etc) , we would need to transform them. And to do that I have used MinMaxScaler of sklearn to scale all the selected features to a given range (i.e between 0 and 1), using the below code .<br>


```python
from sklearn import preprocessing
data = featureFormat(my_dataset, kBest_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
```

## 3. Choosing Different Algorithm and Tuning

### 3.1 Chossing Algorithm

> Q3.What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

I have tried 6 different algorithm and found the naive bayes algorithm works best.Below is the list of all the algorithms i have tried.

    1. Naive Bayes
    2. Decision Tree
    3. Random Forest
    4. Support Vector Machine
    5. k Nearest Neighbours
    6. AdaBoost
  
 During experiment it is found that all the algorithms gives good accuracy.And as the data highly skwed accuracy would not be the best parameter for evaluation metric.It is found that as there are only 18 (POI = 1) out of 144 data it is difficult to identify best algorithm as there can be the case that algorithm have correctly classified many non POIs and due to which it is having higher accuracy.For the better experimental purpose we have tried 30%,40% data as a testing set and 70%,60% data as training set respectively but it is found that 40% testing dataset gives less effective evalution metrics.
     
### 3.2 Tuning Algorithm

> Q4 What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).

In the abstract sense of machine learning, tuning is working with / "learning from" variable data based on some parameters which have been identified to affect system performance as evaluated by some appropriate metric. Tuning the parameters of an algorithm means adjusting the parameters in a certain way to achieve optimal algorithm performance.
Improved performance reveals which parameter settings are more favorable (tuned) or less favorable (untuned).[[3](https://stackoverflow.com/questions/22903267/what-is-tuning-in-machine-learning)].This can be achieved in various ways e.g. a manual guessing or testing method or automatically with GridSearchCV. I have choosen GridSearchCV as my method of tuning. The performance of an algorithm can be measured in a variety of evaluation metrices and the ones I am using are accuracy, precision, or recall.If you don't tune the algorithm well, the performance may deteriorate. The data won't be trained or "learned" well by the algorithm and it won't be able to make proper predictions on new data.

#### 3.2.1 Grid Search for parameter Tuning

Most classifiers in sklearn have a set of predefined parameters which are passed as arguments to the constructor which inturn is responsible for influencing the performance of the algorithm . Tuning the parameters or more commonly know as the hyperparameters of a classifier, means to optimize the values of those parameters so as to make or guide the algorithm to perform to its best extent . This process is called as Hyperparameter Optimization. It's a final step before presenting results. If this step is not done well, it can lead to the model overfitting or model underfitting of the data.


In this project, as I mentioned earlier i have used,GridSearchCV for parameter tuning.GridSearchCV allows us to construct a grid of all the combinations of parameters, tries each combination, and then reports back the best combination/model. So,GridSearchCV from sklearn is used for parameter tuning in the algorithms that had parameters (SVM, Decision Tree, Random forest, KNN etc).

For example in __Decision Tree__ multiple values were given to the multiple parameters.

'criterion':('gini', 'entropy'),<br>
'splitter':('best','random'),<br>
'max_depth':[3,4,5,6],<br>
'min_samples_split':[2,3]<br>


Each algorithm has their own best values of hyperparams and these values are used while making predictions. Below function named tune_param is the function I created for tuning the params for each and every combination of algorithms with their specified parameter values.It prints out the best combinations of params for the corresponding model after performing the tuning for 10 iterations, along with the average evaluation metrics results (accuracy, precision, recall) for the same model.


```python
def tune_params(grid_search, features, labels, params, iters = 10):
    """ given a grid_search and parameters list (if exist) for a specific model,
    along with features and labels list,
    it tunes the algorithm using grid search and prints out the average evaluation metrics
    results (accuracy, percision, recall) after performing the tuning for iter times,
    and the best hyperparameters for the model
    """
    accuracy = []
    precision =[]
    recall = []
    # Precision i.e ratio of true positives out of all positives (true + false)
    # Recall i.e ratio of true positives out of true positives and false negatives
    #p = []
    #r = []
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
        
```

## 4. Validation

> Q5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

In machine learning, model validation is referred to as the process where a trained model is evaluated with a testing data set.[[4](https://link.springer.com/referenceworkentry/10.1007%2F978-1-4419-9863-7_233)].The training dataset and testing dataset were seperate part of the data.The dataset is divided into training and testing set.Then the model is trained using the training datset and using which testing dataset is evaluted.The main purpose of using the testing data set is to test the generalization ability of a trained model.The classic mistake in validation process is to not split our data into training/testing datasets, which often leads to overfitting.Sometimes it happens that model gives excellent accuracy for the training set but it performs poor for the testing dataset.It generally happens because of overfitting,it also happens that algorithm which is trained on particular portion of the dataset and when tested on totaly different type of values of the dataset.

For achieving cross-validation in POI Identifier I have used Cross Validation __train_test_split__ function (found in sklearn.cross_validation) to split 30% of my dataset as for testing. then I used sklearn.metrics accuracy, precision and recall scores to validate my algorithms.


```python
from sklearn.cross_validation import train_test_split
for iteration in range(iters):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, 
                                                                             random_state = iteration)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    accuracy = accuracy + [accuracy_score(labels_test, predicts)]
    precision = precision + [precision_score(labels_test, predicts)]
    recall = recall + [recall_score(labels_test, predicts)]
```

## 5. Evaluation Metrics

> Q6. Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

Various evaluation metrics used were:<br>

Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances,while recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.[[5](https://en.wikipedia.org/wiki/Precision_and_recall)].

- __Precision__ :Precision is the ratio of true positives out of true positive and true negatives.
- __Recall__ :Recall is the ratio of true positives out of true positives and false negatives.
- __Accuracy__ :Accuracy is the degree to which the result of a measurement, calculation, or specification conforms to the correct value or a standard.

For example, consider that our algorithm is found to have a precision score of 0.30 and a recall score of 0.35. This essentially means that, if we are to use this model and it makes a predict for 100 POIs, there would be 35 people that are actually POIs and the rest 65 would be Non-POIs.Also,this model can find 30% of all real(true positive) POIs in its prediction.



```python
import pandas as pd

#For Example consider below table

Table_confusion = pd.DataFrame()
column_1=['**Classified**','**0**','**1**']
column_2 =['**0**','35','15']
column_3=['**1**','10','40']

Table_confusion['column 1']=column_1
Table_confusion['column 2']=column_2
Table_confusion['column 3']=column_3


print("\n Confusion Matrix\n")
pandas_df_to_markdown_table(Table_confusion)
```

    
     Confusion Matrix
    
    


column 1|column 2|column 3
---|---|---
**Classified**|**0**|**1**
**0**|35|10
**1**|15|40



Now consider the above table 35 on [0,1] meaning that there are 35 people who were not in POI and also correctly classfied as not POI.<br>
Now value 40 on diagonal [1,0] is the no of people who were POI(Person of Interest) and correctly classified as POI.
<br>
In this  case<br>
Precision = 35/(35+15)=0.70<br>
Recall =35/(35+10)=0.77<br>
Accuracy = (35+45)/(35+40+15+10)=0.80

## Algorithm Performance

I finally selected Decision Tree  as my algorithm after trying many others. Performance of the final algorithm selected (via Tester.py) is below:

__Final Results<br>__
GaussianNB(priors=None)<br>
&nbsp;&nbsp;&nbsp;&nbsp;  Accuracy: 0.82707<br>
&nbsp;&nbsp;&nbsp;&nbsp;  Precision: 0.35000<br>
&nbsp;&nbsp;&nbsp;&nbsp;  Recall: 0.34650<br>
&nbsp;&nbsp;&nbsp;&nbsp;  F1: 0.34824<br>
&nbsp;&nbsp;&nbsp;&nbsp;  F2: 0.34719<br>
&nbsp;&nbsp;&nbsp;&nbsp;  Total predictions: 15000<br>
&nbsp;&nbsp;&nbsp;&nbsp;  True positives:  693<br>
&nbsp;&nbsp;&nbsp;&nbsp;  False positives: 1287<br>
&nbsp;&nbsp;&nbsp;&nbsp;  False negatives: 1307<br>
&nbsp;&nbsp;&nbsp;&nbsp;  True negatives: 11713<br>

As the method which is used in tester.py is more robust i choose the best algorithm based on that result.
But with that i have also shown results of my tuning with old features and new features below.By using that i got Adaboost as the best model.

I have also attached Table for individual algorithm 



## Tuning of Various Algorithm

### 1. Tuning of Decision Tree


```python
from sklearn import tree
t0=time()
dt_clf = tree.DecisionTreeClassifier()
dt_param = {'criterion':('gini', 'entropy'),
            'splitter':('best','random'),
            'max_depth':[3,4,5,6],
            'min_samples_split':[2,3]
           }
#we have also given max_depth to limited value as we dont want to overfit the tree. 
#we have given min_samples_split : 2,3 as there are only few poi=1 in dataset so 
#using low min_samples_split we can classify.
dt_grid_search = GridSearchCV(estimator = dt_clf, param_grid = dt_param)

print("Decision Tree model evaluation")
tune_params(dt_grid_search, features, labels, dt_param)
print("Processing time:",round(time()-t0,3),"s")

tune_params(dt_grid_search,new_features,new_labels,dt_param)

#we can also count same for new parameters copying same and counting individually!
```

### Results for Decision Tree


```python
Old Features:

Decision Tree model evaluation
accuracy: 0.85
precision: 0.322619047619
recall: 0.183293650794
Confusion Matrix:
[[49  3]
[ 5  1]]

min_samples_split = 2,
splitter = random,
criterion = entropy,
max_depth = 3,
('Processing time:', 12.992, 's')

New Features :
accuracy: 0.843103448276
precision: 0.355952380952
recall: 0.0961544011544
Confusion Matrix:
[[46  6]
[ 5  1]]

min_samples_split = 3, 
splitter = best, 
criterion = entropy,
max_depth = 3, 
('Processing time:', 12.888, 's')


```

### 2. Tuning of Support Vector Machine


```python
from sklearn import svm
t0=time()
svm_clf = svm.SVC()
svm_param = {'kernel':('linear', 'rbf', 'sigmoid'),
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'C': [0.1, 1, 10, 100, 1000]
            }
svm_grid_search = GridSearchCV(estimator = svm_clf, param_grid = svm_param)
print("SVM model evaluation")
tune_params(svm_grid_search, features, labels, svm_param)
print("Processing time:",round(time()-t0,3),"s")

tune_params(svm_grid_search,new_features,new_labels,svm_param)
```

### Results for SVM


```python
Old Features

SVM model evaluation
accuracy: 0.862068965517
precision: 0.233333333333
recall: 0.0953571428571
Confusion Matrix:
[[52  0]
[ 6  0]]

kernel = linear, 
C = 0.1, 
gamma = 1, 
('Processing time:', 31.035, 's')

New features

SVM model evaluation
accuracy: 0.863793103448
precision: 0.275
recall: 0.0992532467532
Confusion Matrix:
[[52  0]
[ 6  0]]

kernel = linear, 
C = 0.1, 
gamma = 1, 
('Processing time:', 25.986, 's')



```

### 3. Tuning of Random Forest


```python
t0 =time()
rf_clf = RandomForestClassifier(n_estimators=10)
#n_estimators = no of trees in the forest
rf_param = {'criterion':('gini','entropy'),
            'max_depth':[3,4,5,6],
            'min_samples_split':[2,3]
            }
rf_grid_search = GridSearchCV(estimator = rf_clf, param_grid = rf_param)

print("Random Forest model evaluation")
tune_params(rf_grid_search, features, labels, rf_param)
print("Processing time:",round(time()-t0,3),"s")
tune_params(rf_grid_search,new_features,new_labels,rf_param)

```

### Results for Random Forest


```python
Old features

Random Forest model evaluation
accuracy: 0.860344827586
precision: 0.431666666667
recall: 0.121626984127
Confusion Matrix:
[[49  3]
[ 4  2]]

min_samples_split = 3, 
criterion = gini, 
max_depth = 3, 
('Processing time:', 75.505, 's')

New Features

Random Forest model evaluation
accuracy: 0.856896551724
precision: 0.308333333333
recall: 0.103300865801
Confusion Matrix:
[[46  6]
 [ 4  2]]

min_samples_split = 3, 
criterion = entropy, 
max_depth = 4, 
('Processing time:', 75.595, 's')
```

### 4. Tuning of KNN


```python
print("\nK Neighbors Execution and Evaluation\n")
t0=time()
k=np.arange(10)+1
knn_clf = KNeighborsClassifier()
knn_param ={'n_neighbors':k}
knn_grid_search  = GridSearchCV(estimator =knn_clf,param_grid =knn_param)
print("KNN evaluation")
tune_params(knn_grid_search, features, labels, knn_param)
print("Processing time:",round(time()-t0,3),"s")

tune_params(knn_grid_search,new_features,new_labels,knn_param)

```

### Results for KNN


```python
Old Features

K Neighbors Execution and Evaluation
KNN evaluation
accuracy: 0.860344827586
precision: 0.133333333333
recall: 0.0553571428571
Confusion Matrix:
[[52  0]
 [ 6  0]]
n_neighbors = 10, 
('Processing time:', 4.417, 's')

New Features
KNN evaluation
accuracy: 0.851724137931
precision: 0.0
recall: 0.0
Confusion Matrix:
[[48  4]
 [ 6  0]]
n_neighbors = 7, 
('Processing time:', 4.315, 's')

```

### 5. Tuning of Adaboost


```python
print("\nAdaBoost Execution and Evaluation\n")
t0=time()
ada_clf = AdaBoostClassifier(learning_rate =1.5,n_estimators = 9, algorithm ='SAMME.R')
ada_param = {}
ada_grid_search = GridSearchCV(estimator = ada_clf,param_grid =ada_param)
tune_params(ada_grid_search, features, labels, ada_param)
print("Processing time:",round(time()-t0,3),"s")

tune_params(ada_grid_search,new_features,new_labels,ada_param)
```

### Results for Adaboost


```python
For old Features

AdaBoost Execution and Evaluation
accuracy: 0.851724137931
precision: 0.517838827839
recall: 0.31386002886
Confusion Matrix:
[[42 10]
 [ 3  3]]

('Processing time:', 4.193, 's')

For New Features

AdaBoost Execution and Evaluation
accuracy: 0.862068965517
precision: 0.549880952381
recall: 0.376756854257
Confusion Matrix:
[[45  7]
 [ 3  3]]

('Processing time:', 3.78, 's')
```


```python
import pandas as pd

Table_df_old = pd.DataFrame()
Table_df_new = pd.DataFrame()
Table_df_tester = pd.DataFrame()


#All values below are programatically generated. These don't include values from Tester.py
Classifier = ['Decision Tree','Support Vector Machine','Gaussian Naive Bayes','Random Forest','K Nearest Neighbors','AdaBoost']

Execution_Time_old = ['12.992','31.035','0.437','75.505','4.417','4.193']
Execution_Time_new = ['12.888','25.986','0.516','75.595','4.315','3.780']



HyperParameters_old = ['min_samples_split = 2, splitter = random, criterion = gini, max_depth = 6' ,
                        'kernel = linear,C = 0.1,gamma = 1',
                       '-',
                        'min_samples_split = 3,criterion = gini,max_depth = 3',
                       'n_neighbors = 10','-']

HyperParameters_new = ['min_samples_split = 3, splitter = best,criterion = entropy,max_depth = 3',
                       'kernel = linear,C = 0.1,gamma = 1',
                       '-',
                       'min_samples_split = 3,criterion = entropy,max_depth = 4',
                       'n_neighbors = 7','-']

Accuracy_old = ['0.85','0.86','0.84','0.86','0.86','0.85']
Accuracy_new = ['0.84','0.86','0.84','0.85','0.85','0.86']
Accuracy_tester = ['0.82','0.86','0.84','0.86','0.87','0.84']

Precision_old = ['0.32','0.23','0.38','0.43','0.13','0.51']
Precision_new = ['0.35','0.28','0.36','0.31','0.0','0.54']
Precision_tester =['0.35','0.27','0.38','0.41','0.62','0.37']
                

Recall_old = ['0.18','0.09','0.33','0.12','0.05','0.31']
Recall_new = ['0.10','0.10','0.33','0.10','0.0','0.37']
Recall_tester = ['0.34','0.10','0.32','0.15','0.17','0.26']

# For Old Unaltered Feature list
Table_df_old ['Classifier']=Classifier
Table_df_old ['HyperParameters']=HyperParameters_old
Table_df_old['Execution Time']=Execution_Time_old
Table_df_old ['Accuracy']=Accuracy_old
Table_df_old ['Precision']=Precision_old
Table_df_old ['Recall']=Recall_old

# For Newly altered Feature list
Table_df_new['Classifier']=Classifier
Table_df_new['HyperParameters']=HyperParameters_new
Table_df_new['Execution Time']=Execution_Time_new
Table_df_new['Accuracy']=Accuracy_new
Table_df_new['Precision']=Precision_new
Table_df_new['Recall']=Recall_new


# For Tester result

Table_df_tester['Classifier']=Classifier
Table_df_tester['Accuracy']=Accuracy_tester
Table_df_tester ['Precision']=Precision_tester
Table_df_tester['Recall']=Recall_tester

def pandas_df_to_markdown_table(df):
    from IPython.display import Markdown, display
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    display(Markdown(df_formatted.to_csv(sep="|", index=False)))

print("\nFor Old Unaltered Feature list\n")
pandas_df_to_markdown_table(Table_df_old)
print("\nFor Newly altered Feature list\n")
pandas_df_to_markdown_table(Table_df_new)
print("\n For tester program with k (10) best features\n")
pandas_df_to_markdown_table(Table_df_tester)
```

    
    For Old Unaltered Feature list
    
    


Classifier|HyperParameters|Execution Time|Accuracy|Precision|Recall
---|---|---|---|---|---
Decision Tree|min_samples_split = 2, splitter = random, criterion = gini, max_depth = 6|12.992|0.85|0.32|0.18
Support Vector Machine|kernel = linear,C = 0.1,gamma = 1|31.035|0.86|0.23|0.09
Gaussian Naive Bayes|-|0.437|0.84|0.38|0.33
Random Forest|min_samples_split = 3,criterion = gini,max_depth = 3|75.505|0.86|0.43|0.12
K Nearest Neighbors|n_neighbors = 10|4.417|0.86|0.13|0.05
AdaBoost|-|4.193|0.85|0.51|0.31



    
    For Newly altered Feature list
    
    


Classifier|HyperParameters|Execution Time|Accuracy|Precision|Recall
---|---|---|---|---|---
Decision Tree|min_samples_split = 3, splitter = best,criterion = entropy,max_depth = 3|12.888|0.84|0.35|0.10
Support Vector Machine|kernel = linear,C = 0.1,gamma = 1|25.986|0.86|0.28|0.10
Gaussian Naive Bayes|-|0.516|0.84|0.36|0.33
Random Forest|min_samples_split = 3,criterion = entropy,max_depth = 4|75.595|0.85|0.31|0.10
K Nearest Neighbors|n_neighbors = 7|4.315|0.85|0.0|0.0
AdaBoost|-|3.780|0.86|0.54|0.37



    
     For tester program with k (10) best features
    
    


Classifier|Accuracy|Precision|Recall
---|---|---|---
Decision Tree|0.82|0.35|0.34
Support Vector Machine|0.86|0.27|0.10
Gaussian Naive Bayes|0.84|0.38|0.32
Random Forest|0.86|0.41|0.15
K Nearest Neighbors|0.87|0.62|0.17
AdaBoost|0.84|0.37|0.26



## References:

[[1](https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475461/lessons/3174288624239847/concepts/31803986370923)] : Udacity Machine Learning Project : "Identify Fraud from Enron Email" Description <br>
[[2](https://en.wikipedia.org/wiki/Feature_scaling)]: Feature Scaling, From Wikipedia, the free encyclopedia<br>
[[3](https://stackoverflow.com/questions/22903267/what-is-tuning-in-machine-learning)]: Tuning in machine learning<br>
[[4](https://link.springer.com/referenceworkentry/10.1007%2F978-1-4419-9863-7_233)] : Springer article on Validation<br>
[[5](https://en.wikipedia.org/wiki/Precision_and_recall)] : Precision and recall, From Wikipedia, the free encyclopedia<br>
[[6](https://github.com/jasminej90/dand5-identity-fraud-from-enron-email)]: Github : Identity fraud from Enron Email By Jasmin J
