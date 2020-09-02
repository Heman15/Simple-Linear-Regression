import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
area = np.array([200,300,400,600,800]).reshape(-1,1)
price = np.array([1100,1300,1500,1700,5000]).reshape(-1,1)
reg = LinearRegression()
reg.fit(area,price)
y_pr = reg.predict(area)
slope = reg.coef_
intercept = reg.intercept_
y_p = reg.intercept_* area + reg.coef_
plt.scatter(area, price)
plt.plot(area,y_p, color='red')
plt.xlabel('area')
plt.ylabel('price' )
plt.show()
print(slope)
print (intercept)
print(y_pr)



