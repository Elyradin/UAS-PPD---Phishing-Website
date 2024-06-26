import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib


def train_logistic_regression(X, y, filename='logistic_regression_model.model'):
    # Initialize logistic regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X, y)

    filename = 'model_phishing.model'

    # Save the trained model
    joblib.dump(model, filename)
    
if __name__ == '__main__':
    #read the dataset
    pd.options.display.max_columns=None
    df_ori = pd.read_csv('E:/UAS PPD - Phishing Website/create_model/web-page-phishing.csv')
    df_ori = df_ori.dropna()
    #get the X and y
    df_X = df_ori.drop(['phishing'],axis=1)
    df_y = df_ori['phishing']
    #relabelling the class
    df_y=df_y.replace(0, 'not phishing')
    df_y=df_y.replace(1, 'phishing')
    
    #generate the model 
    print('Generate Decision Tree Model')
    train_logistic_regression(df_X, df_y)