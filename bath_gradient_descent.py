# # Imports

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

## Load in the Data and Examine

# Read in data into a dataframe 
data = pd.read_csv('C:/Users/User/Downloads/Spring 2019/ML/Assignment1/Features.csv')

## No of observations and variables
data.shape


# Display top of dataframe
data.head()

# # Data Types and Missing Values

# See the column data types and non-missing values
data.info() ## No missing values

# Statistics for each column
data.describe()

## Boxplot to identify outliers
import seaborn as sns
sns.set(style="whitegrid")
# data = sns.load_dataset(data)
## ax = sns.boxplot(x="day", y="total_bill", data=tips)
ax = sns.boxplot(x=data["derived_4"])

## Identifying specific outliers using Z score > 3 as threshold
from scipy import stats
z = np.abs(stats.zscore(data["derived_4"]))
print(np.where(z > 3))

# # Correlations between Features and Target

# Find all correlations and sort 
correlations_data = data.corr()['output'].sort_values()
#print(correlations_data)

# Print the most negative correlations
#print(correlations_data.head(20), '\n')

# Print the most positive correlations
print(correlations_data.tail(16)) 

## Correlation between variables
# Matplotlib visualization
import matplotlib.pyplot as plt 
correlation = data.corr()
plt.figure(figsize=(54,54))
sns.heatmap(correlation, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},
            xticklabels=correlation.columns.values,
            yticklabels=correlation.columns.values)
            

newdata = data[['page_mentions', 'derived_5', 'derived_25', 'derived_13', 'derived_18', 'derived_8', 'comments_24_post', 'comments_diff', 'comments_counts_24', 'post_share_count', 'H_hours', 'output']]
newdata.describe()

## Correlation between variables for new dataset

correlation2 = newdata.corr()
plt.figure(figsize=(16,16))
sns.heatmap(correlation2, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
            xticklabels=correlation2.columns.values,
            yticklabels=correlation2.columns.values)
           
ax = sns.boxplot(x=newdata["derived_5"])

bx = sns.boxplot(x=newdata["page_mentions"])
z = np.abs(stats.zscore(data["page_mentions"]))
print(np.where(z > 3))

### Split Into Training and Testing Sets
from sklearn.model_selection import train_test_split

## Separate out the features and targets
features = newdata.drop(columns='output')
targets = pd.DataFrame(newdata['output'])

# Split into 70% training and 30% testing set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# # Feature Scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.shape

X_test.shape

# Convert y to one-dimensional array (vector)
y_train = np.array(y_train).reshape((-1, ))
y_test = np.array(y_test).reshape((-1, ))

y_test.shape

#defining hypothesis, bath gradient descent algorithm and linear regression model 
def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    return h

def BGD(theta, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/X.shape[0]) * sum((h-y) * X.transpose()[j])
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y_train))
        print("Iteration %d | Cost: %f" % (i, cost[i]))
    theta = theta.reshape(1,n+1)
    return theta, cost

def linear_regression(X, y, alpha, num_iters):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    # initializing the parameter vector...
    theta = np.zeros(n+1)
    # hypothesis calculation....
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent...
    theta, cost = BGD(theta,alpha,num_iters,h,X,y,n)
    return theta, cost

# calling the principal function with learning_rate = 0.0001 and 
# num_iters = 300000
theta, cost = linear_regression(X_train, y_train, 0.0001, 300)
theta

#plotting cost vs no of iterations to analyze the cost function behaviour
import matplotlib.pyplot as plt
cost = list(cost)
n_iterations = [x for x in range(1,301)]
plt.plot(n_iterations, cost)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')

#predicting no of facebook comments using theta values
one_column = np.ones((X_test.shape[0],1))
X_test = np.concatenate((one_column, X_test), axis = 1)
X_test.shape
xTrans = X_test.transpose()
xTrans
predictions = np.dot(theta,xTrans)
predictions
