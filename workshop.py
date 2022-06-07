import pandas as pd
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt

# DATASET PREPARATION & INTEGRATION
# go to the repository: https://github.com/MikiKru/PWITS

# take the url of file with data -> energy_profile.csv
file_url = "https://raw.githubusercontent.com/MikiKru/PWITS/master/energy_profile.csv"

# get data from file to DataFrame object
energy_profile_df = pd.read_csv(file_url, sep=",")

# print first 10 rows
energy_profile_df.head()

# DATASET PREPROCESSING
# select columns to training dataset
X_cols = ['Time_tick', 'PV', 'Demand', 'Day', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'AvgTemp_F']

# split into train and test datasets -> target variable Yn_class will be y vector
X_train, X_test, y_train, y_test = train_test_split(energy_profile_df[X_cols], energy_profile_df['Yn_class'], test_size=0.4)

# TRAINING OF MACHINE LEARNING MODEL
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)

# TESTING OF MACHINE LEARNING MODEL
y_pred = clf.predict(X_test)

# FUNCTION THAT GENERATES RESULTS

def get_results(clf, X, y):
  plot_confusion_matrix(clf, X_train, y_train)
  plt.title("TRAIN:" + str(clf))
  plt.show()
  plot_roc_curve(clf, X_test, y_test)
  plt.show()

  print("ACC", accuracy_score(y_train,y_pred_train))
  print("PREC", precision_score(y_train,y_pred_train))
  print("RECALL", recall_score(y_train,y_pred_train))
  print("AUC", roc_auc_score(y_train,y_pred_train))

get_results(clf, X_test, y_test)