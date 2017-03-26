import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Tkinter as Tk
import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

from sklearn import metrics


#reading the data from csv file to data frame of pandas

df=pd.read_csv("/home/nikhilponnuru/Desktop/project-datascience/trainingdata.csv")

print(df.head(10))  #used to display top 10 rows
print (df.describe())

print(df['Married'].value_counts())   #gives count of all types individually inside Married e.g:- no of married and no of unmarried


#df['ApplicantIncome'].hist(bins=50)

#df.boxplot(column='ApplicantIncome')
#df.boxplot(column='ApplicantIncome', by = 'Education')
#df['LoanAmount'].hist(bins=50)


temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print 'Frequency Table for Credit History:'
print temp1

print '\nProbility of getting loan for each Credit History class:'
print temp2



fig = plt.figure(figsize=(1,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")


#displying 2 graphs into one
temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

#Check missing values in the dataset

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

df['Self_Employed'].fillna('No', inplace=True)

print(df.apply(lambda x: sum(x.isnull()), axis=0))

table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)

#df['LoanAmount'].hist(bins=20)

df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)



plt.show()




# making of a predictive model



def classification_model(model, data, predictors, outcome):
   
    model.fit(data[predictors], data[outcome])

   
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print "Accuracy : %s" % "{0:.3%}".format(accuracy)

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        
        train_target = data[outcome].iloc[train]

       
        model.fit(train_predictors, train_target)

      
        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

   
    model.fit(data[predictors], data[outcome])

      #regression analysis

      
outcome_var = df['Loan_Status']
model = LogisticRegression()
predictor_var = df['Credit_History']
classification_model(model, df,predictor_var,outcome_var)


