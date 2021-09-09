
# An Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing
# advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home
# and order either on a mobile app or website for the clothes they want.

# The company is trying to decide whether to focus their efforts on their mobile app experience or their website.



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ** Reading Ecommerce Customers csv file as a DataFrame called customers.**

customers = pd.read_csv('Ecommerce Customers')


# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member.


print(customers.head())

print(customers.describe())

print(customers.info())


# **Exploratory Data Analysis (EDA)

# **Jointplot to compare the Time on Website and Yearly Amount Spent columns.**

sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
plt.show()


#Does the correlation make sense?

#NO
print(customers.corr())


# **Jointplot to compare the Time on App and Yearly Amount Spent columns.**

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
plt.show()


# **Jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex')
plt.show()


# **Pairplot to explore these types of relationships across the entire data set.

sns.pairplot(customers)
plt.show()

# **Based off Pairplot what looks to be the most correlated feature with Yearly Amount Spent?**

#Length of Membership


# **Linear model plot of  Yearly Amount Spent vs. Length of Membership. **

sns.lmplot(x='Yearly Amount Spent', y='Length of Membership', data=customers)
plt.show()


# ## Training and Testing Data by using Linear Regression

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)


# **Coefficients of the model**

print(lm.coef_)


# ## Predicting Test Data

prediction = lm.predict(X_test)


# ** Scatterplot of the real test values versus the predicted values. **

plt.scatter(y_test, prediction)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# ## Evaluating the Model

from sklearn import metrics

print("MAE: ", metrics.mean_absolute_error(y_test, prediction))
print("MSE: ", metrics.mean_squared_error(y_test, prediction))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, prediction)))



# Model with a good fit.

# **Histogram of the residuals and make sure it looks normally distributed.

sns.distplot(y_test-prediction, bins=50)
plt.show()


# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website
# development? Or maybe that doesn't even really matter, and Membership Time is what is really important.

# Let's interpret the coefficients table to get an idea.

coefficients_table = pd.DataFrame(lm.coef_, X.columns, columns=['coeff'])
print(coefficients_table)


# ** Insights from coefficients table **

# *Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of
# 25.98 total dollars spent.

# *Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of
# 38.59 total dollars spent.

# *Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of
# 0.19 total dollars spent.

# *Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of
# 61.27 total dollars spen


# **The company should focus more on their mobile app or on their website?**

# This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the
# mobile app, or develop the app more since that is what is working better. This sort of answer really depends on the
# other factors going on at the company, you would probably want to explore the relationship between Length of
# Membership and the App or the Website before coming to a conclusion!
