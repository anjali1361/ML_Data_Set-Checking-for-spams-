import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

##Step1: Load Dataset
dataframe = pd.read_csv("spam.csv")#the type of object that panda returns to us is called dataframe
print(dataframe.describe())#describing data in terms of freequency count,top,unique..

##Step2: Split in to Training and Test Data

x = dataframe["EmailText"]
y = dataframe["Label"]

x_train,y_train = x[0:4457],y[0:4457]#we separated our data as training data set
x_test,y_test = x[4457:],y[4457:]#we separated our data as test data set

##Step3: Extract Features
cv = CountVectorizer()  
features = cv.fit_transform(x_train)

##Step4: Build a model
tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}

model = GridSearchCV(svm.SVC(), tuned_parameters)

model.fit(features,y_train)

print(model.best_params_)
#Step5: Test Accuracy
print("Accuracy:"+str(model.score(cv.transform(x_test),y_test)))



