Index(['7', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8',
       ...
       '0.658', '0.659', '0.660', '0.661', '0.662', '0.663', '0.664', '0.665',
       '0.666', '0.667'],
      dtype='object', length=785)

import matplotlib.pyplot as plt

# Extract data to plot
x = df.iloc[:, 0] # First column
y = df.iloc[:, 10] # Second column

# Create plot
plt.plot(x, y)

# Customize plot
plt.title("My Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# Display plot
plt.show()


     


 import pandas as pd
 import matplotlib.pyplot as plt

 path = '/content/sample_data/mnist_test.csv'
df = pd.read_csv(path)
cols = df.columns[:2]

# Create scatter plot of the first two columns
plt.scatter(df[cols[0]][:100], df[cols[1]][:100])

# Set axis labels
plt.xlabel(cols[0])
plt.ylabel(cols[1])

# Display plot
plt.show()
     



 import matplotlib.pyplot as plt

 path = '/content/sample_data/mnist_test.csv'
df = pd.read_csv(path)
new = df.columns
# Print column names
print(new)
     
Index(['7', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8',
       ...
       '0.658', '0.659', '0.660', '0.661', '0.662', '0.663', '0.664', '0.665',
       '0.666', '0.667'],
      dtype='object', length=785)

import seaborn as sns
sns.scatterplot(data=df, x='7', y='0.658')

plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.show()
     


import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# plot the data as a line graph
ax.plot(df['7' ], df['0.658'])

# set the axis labels and title
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Line Graph Example')

# display the plot
plt.show()
     


fig, ax = plt.subplots()

# plot the data as a bar graph
ax.bar(df['7'], df['0.658'])

# set the axis labels and title
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Bar Graph Example')

# display the plot
plt.show()
     


fig, ax = plt.subplots()

# plot the data as a histogram
ax.hist(df['7'], bins=20)

# set the axis labels and title
ax.set_xlabel('Data Values')
ax.set_ylabel('Frequency')
ax.set_title('Histogram Example')

# display the plot
plt.show()


     


fig, ax = plt.subplots()

# plot the data as a scatter plot
ax.scatter(df['7'], df['0.658'])

# set the axis labels and title
ax.set_xlabel('X Values')
ax.set_ylabel('Y Values')
ax.set_title('Scatter Plot Example')

# display the plot
plt.show()
     


import pandas as pd
import matplotlib.pyplot as plt

# load data from CSV file
df = pd.read_csv('/content/sample_data/mnist_test.csv', index_col=0)

fig, ax = plt.subplots()

# plot the data as a heat map
im = ax.imshow(df.values, cmap='YlOrRd')

# create a colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# set the axis labels and title
ax.set_xticks(np.arange(df.shape[7]))
ax.set_yticks(np.arange(df.shape[7]))
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.index)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Heat Map Example')

# loop over data dimensions and create text annotations
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        text = ax.text(j, i, f'{df.values[i, j]:.2f}', ha='center', va='center', color='black')

# display the plot
plt.show()
     
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
<ipython-input-34-e1b1b10c834f> in <cell line: 16>()
     14 
     15 # set the axis labels and title
---> 16 ax.set_xticks(np.arange(df.shape[7]))
     17 ax.set_yticks(np.arange(df.shape[7]))
     18 ax.set_xticklabels(df.columns)

IndexError: tuple index out of range


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data from CSV file
df = pd.read_csv('/content/sample_data/mnist_test.csv')
x = df['7'].values
y = df['0.658'].values

# create a 2D array of random values
z = np.random.rand(len(x), len(y))

# create a new figure and axis
fig, ax = plt.subplots()

# plot the data as a heat map
im = ax.imshow(z, cmap='YlOrRd')

# set the axis labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('2D Heat Map without Z Index')

# set the tick marks and labels
ax.set_xticks(np.arange(0, len(x), 1))
ax.set_yticks(np.arange(0, len(y), 1))
ax.set_xticklabels(x)
ax.set_yticklabels(y)

# display the plot
plt.show()
     

seaborn


import seaborn as sns
import pandas as pd

# load data from csv file
data = pd.read_csv('/content/sample_data/mnist_test.csv')

# create a line plot
sns.lineplot(x='7', y='0.658', data=data)
     
<Axes: xlabel='7', ylabel='0.658'>

bar graph


sns.barplot(x='7', y='0.658', data=data)
     
<Axes: xlabel='7', ylabel='0.658'>

histogram


sns.distplot(data['7'], kde=False)
     
<ipython-input-54-b3623f04a57a>:1: UserWarning: 

`distplot` is a deprecated function and will be removed in seaborn v0.14.0.

Please adapt your code to use either `displot` (a figure-level function with
similar flexibility) or `histplot` (an axes-level function for histograms).

For a guide to updating your code to use the new functions, please see
https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751

  sns.distplot(data['7'], kde=False)
<Axes: xlabel='7'>

Scatter plots


sns.scatterplot(x='7', y='0.658', data=data)
     
<Axes: xlabel='7', ylabel='0.658'>

heat map


sns.heatmap(data.corr())
     
<Axes: >
