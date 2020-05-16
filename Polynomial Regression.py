#polynomial regression
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

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(x)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
X_grid=np.arange(min(x),max(x),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting a new result with Linear regression
lin_reg.predict([[6.5]])

#Predicting a new result with polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
