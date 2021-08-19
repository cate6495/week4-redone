import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Salary.csv")

x = df.iloc[:, :1].values
y = df.iloc[:, 1:].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

pickle.dump(regressor, open("model.pkl", "wb"))