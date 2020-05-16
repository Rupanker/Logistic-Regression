#Random Forest Regression
#IMPORT data libraries
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd 

#import the dataset
dataset=pd.read_csv("Position_Salaries.csv")
x= dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#splitting the dataset into the Training set and Test Set
'''from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)'''

#feature scalling
'''from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)'''

# Fitting the Random Forest Regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)

#predictiing a new result
y_pred=regressor.predict([[6.5]])

# Visualising the Random Forest Regression results(for higher resolution and smoother curve)
X_grid=np.arange(min(x),max(x),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()