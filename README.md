# project2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
 
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data.head()
print("Shape of data is", data.shape)
print("="*50)

data.info()
data.isnull().sum()
data.duplicated().sum()
data.hist(figsize=(15, 10))
plt.show()
corr = data.corr()
corr
plt.figure(figsize=(8,5))
sns.heatmap(corr, cmap='YlGnBu', annot=True, vmin=-1, vmax=1)
plt.title('Relation between features and Diabetes')
plt.show()
