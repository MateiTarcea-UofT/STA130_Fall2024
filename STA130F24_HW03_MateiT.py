#!/usr/bin/env python
# coding: utf-8

# # HW Assignment 3 - Matei Tarcea
# First chat log with ChatGPT (Q 1-4): https://chatgpt.com/share/66f4ca19-2268-8006-8a1f-35a6c412086c
# Second chat log with ChatGPT (Q5): https://chatgpt.com/share/66f6252a-d9a0-8006-a9d2-c13ed3e8278a
# Third chat log with ChatGPT (Q6): https://chatgpt.com/share/66f62549-36e4-8006-b1b4-16418e8c3a88

# In[32]:


import pandas as pd 
pingees = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
pingees


# In[22]:


#https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()


# # Question 1:

# In[45]:


import plotly.express as px
import plotly.graph_objects as go

pingees = pingees.dropna(subset=['species', 'flipper_length_mm'])

# Create a separate figure for each species
species_list = pingees['species'].unique()

for species in species_list:
    # Filter data for the current species
    species_data = pingees[pingees['species'] == species]
    
    # Calculate statistics
    flipper_lengths = species_data['flipper_length_mm']
    mean = flipper_lengths.mean()
    median = flipper_lengths.median()
    std = flipper_lengths.std()
    q1 = flipper_lengths.quantile(0.25)
    q3 = flipper_lengths.quantile(0.75)
    min_val = flipper_lengths.min()
    max_val = flipper_lengths.max()
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
    
    # Create a histogram using plotly express
    fig = px.histogram(species_data, x='flipper_length_mm', nbins=30, 
                       title=f'Flipper Length Distribution for {species}',
                       labels={'flipper_length_mm': 'Flipper Length (mm)'},
                       opacity=0.75)

    # Convert to a graph object figure to add lines and rectangles
    fig = go.Figure(fig)
    
    # Add mean and median lines
    fig.add_vline(x=mean, line=dict(color='blue', width=2), annotation_text="Mean", annotation_position="top left")
    fig.add_vline(x=median, line=dict(color='green', width=2, dash='dash'), annotation_text="Median", annotation_position="top right")
    
    # Add range (min to max) rectangle
    fig.add_vrect(x0=min_val, x1=max_val, fillcolor="lightgrey", opacity=0.2, line_width=0, annotation_text="Range", annotation_position="top left")
    
    # Add IQR rectangle
    fig.add_vrect(x0=q1, x1=q3, fillcolor="yellow", opacity=0.3, line_width=0, annotation_text="IQR", annotation_position="top left")
    
    # Add ±2 standard deviations from the mean rectangle
    fig.add_vrect(x0=lower_bound, x1=upper_bound, fillcolor="red", opacity=0.1, line_width=0, annotation_text="±2 Std", annotation_position="top left")
    
    # Update layout
    fig.update_layout(xaxis_title='Flipper Length (mm)', yaxis_title='Count')

    # Show the figure
    fig.show(renderer = 'png')


# # Question 2

# In[47]:


import seaborn as sns
import matplotlib.pyplot as plt

for species in species_list:
    
    species_data = pingees[pingees['species'] == species]['flipper_length_mm']
    
    # Calculate statistics
    mean = species_data.mean()
    median = species_data.median()
    std = species_data.std()
    q1 = species_data.quantile(0.25)
    q3 = species_data.quantile(0.75)
    min_val = species_data.min()
    max_val = species_data.max()
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
 # Create the KDE plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(species_data, fill=True, color='skyblue', label=f'{species} Flipper Length KDE')
    
    # Add vertical lines for mean and median
    plt.axvline(mean, color='blue', linestyle='-', linewidth=2, label='Mean')
    plt.axvline(median, color='green', linestyle='--', linewidth=2, label='Median')
    
    # Shade range (min to max)
    plt.axvspan(min_val, max_val, color='grey', alpha=0.2, label='Range')
    
    # Shade IQR
    plt.axvspan(q1, q3, color='yellow', alpha=0.3, label='IQR')
    
    # Shade ±2 standard deviations from the mean
    plt.axvspan(lower_bound, upper_bound, color='red', alpha=0.1, label='±2 Std')
    
    # Add labels and legend
    plt.title(f'Flipper Length KDE for {species}', fontsize=14)
    plt.xlabel('Flipper Length (mm)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper left')
    
    # Show the plot
    plt.show()


# # Question 3: 
# I told chatGPT that I believed that box plots were good at showing outliers, histograms for trends, and KDE's for showing denseties in data. Chat agreed with me and expanded on my points. 
# 
# # Chat Response: 
# 
# Box Plots:
# 
# Strength: Excellent at displaying the distribution of data in terms of quartiles, and particularly effective at highlighting outliers and the spread of the middle 50% (interquartile range, or IQR).
# Use case: Great for comparing distributions across multiple categories, detecting skewness, and spotting extreme values easily.
# Histograms:
# 
# Strength: Good for showing the frequency distribution of data, giving a clear view of how data is distributed across different bins, which helps in understanding trends and patterns in the data.
# Use case: Ideal when you want to see the general shape of the data distribution, particularly for discrete or grouped continuous variables.
# KDE (Kernel Density Estimation):
# 
# Strength: Great at providing a smooth estimate of the probability density function of a continuous variable, giving a sense of how densely the data points are distributed.
# Use case: Perfect for visualizing the underlying distribution of continuous data in a smooth, readable form, especially when you're more interested in the overall distribution than in the individual counts within bins.
# 
# 
# Each method highlights different aspects of the data, and the choice depends on what aspect of the data you want to focus on. Together, they can provide a comprehensive understanding of the distribution and characteristics of your dataset.
# 
# ChatGPT also listed some cons for each data type. 
# 
# Yes, each data visualization type also has its limitations or drawbacks. Here’s a look at the cons for box plots, histograms, and KDEs:
# 
# 1. Box Plots:
# Con: Lack of detail.
# Box plots are minimalistic and don't show the distribution shape in detail (e.g., whether the data is multimodal or where most of the data points lie within the quartiles).
# Outliers are shown as points but with no further context on how far out they are or how many there are in that region.
# Con: Not ideal for small datasets.
# In small datasets, outliers can be frequent, and the true distribution may not be captured well.
# 2. Histograms:
# Con: Bin sensitivity.
# Histograms depend on the choice of bin width, which can distort trends. Too few bins can oversimplify the data, while too many bins can overcomplicate it, making it hard to detect meaningful patterns.
# Con: Less smooth.
# Histograms give a discrete view of the data, and the blocky nature can sometimes obscure the smooth trends that are present in the data, making the exact shape harder to interpret.
# Con: Can obscure outliers.
# Outliers may be harder to spot since they can blend into a large bin.
# 3. KDE (Kernel Density Estimation):
# Con: Can be misleading with small datasets.
# KDEs can produce a smooth distribution, but this can sometimes mislead by suggesting trends in data that aren’t actually there, especially when the dataset is small.
# Con: Bandwidth sensitivity.
# The smoothness of a KDE depends on the bandwidth parameter. A small bandwidth may create an overly wavy line, while a large bandwidth can overly smooth the distribution, hiding important details.
# Con: No explicit counts.
# Unlike histograms, KDEs don’t directly show how many observations fall within certain intervals or how they are grouped, making it harder to gauge the actual frequency of data points in specific areas.
# Summary of Cons:
# Box plots don’t provide detailed distribution information.
# Histograms can be affected by bin choice and may obscure finer trends or outliers.
# KDEs can be misleading with small data or improper bandwidth and lack explicit count information.
# Understanding the limitations helps in choosing the right visualization tool depending on the dataset and the specific insights you're looking for.
# 
# 

# In[31]:


from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

n = 1500
data1 = stats.uniform.rvs(0, 10, size=n)
data2 = stats.norm.rvs(5, 1.5, size=n)
data3 = np.r_[stats.norm.rvs(2, 0.25, size=int(n/2)), stats.norm.rvs(8, 0.5, size=int(n/2))]
data4 = stats.norm.rvs(6, 0.5, size=n)

fig = make_subplots(rows=1, cols=4)

fig.add_trace(go.Histogram(x=data1, name='A', nbinsx=30, marker=dict(line=dict(color='black', width=1))), row=1, col=1)
fig.add_trace(go.Histogram(x=data2, name='B', nbinsx=15, marker=dict(line=dict(color='black', width=1))), row=1, col=2)
fig.add_trace(go.Histogram(x=data3, name='C', nbinsx=45, marker=dict(line=dict(color='black', width=1))), row=1, col=3)
fig.add_trace(go.Histogram(x=data4, name='D', nbinsx=15, marker=dict(line=dict(color='black', width=1))), row=1, col=4)

fig.update_layout(height=300, width=750, title_text="Row of Histograms")
fig.update_xaxes(title_text="A", row=1, col=1)
fig.update_xaxes(title_text="B", row=1, col=2)
fig.update_xaxes(title_text="C", row=1, col=3)
fig.update_xaxes(title_text="D", row=1, col=4)
fig.update_xaxes(range=[-0.5, 10.5])

for trace in fig.data:
    trace.xbins = dict(start=0, end=10)
    
# This code was produced by just making requests to Microsoft Copilot
# https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk3/COP/SLS/0001_concise_makeAplotV1.md

fig.show(renderer = 'png') # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# # Question 4: 
# 
# 4.1: Graph A and B have similar means and variances.  
# 4.2: Graphs A and D and B and D have similar means but different variances.  
# 4.3: Graphs A and C have similar variances but different means.   
# 4.4 Graphs C and D have quite different variances and quite different means.   

# # ChatBot Summary for pre-class hw: 
# 
# In this interaction, we explored several topics:
# 
# 1. **Plotly Visualizations:**
#    - We discussed using Plotly to visualize the `flipper_length_mm` distribution in the **pingees** dataset (hypothetical replacement of the **penguins** dataset) for each species. We created histograms using **plotly.express** (`px`) and marked statistical metrics like mean, median, range, interquartile range (IQR), and ±2 standard deviations.
# 
# 2. **Seaborn KDE Visualization:**
#    - You wanted to replicate the same visualization using **Seaborn** KDE plots. We overlaid the KDE with vertical lines for mean and median, and shaded regions for the range, IQR, and ±2 standard deviations.
# 
# 3. **Pros of Different Data Visualizations:**
#    - You shared your views on the strengths of different visualizations:
#      - **Box plots** are good at showing outliers and the main range of data.
#      - **Histograms** are effective at displaying trends in data.
#      - **KDEs** are excellent for showing data densities.
#    - I agreed with your assessment and added some additional details about each method’s strengths.
# 
# 4. **Cons of Different Data Visualizations:**
#    - We then discussed the limitations of each method:
#      - **Box plots** lack detailed distribution information and struggle with small datasets.
#      - **Histograms** can be sensitive to bin choice, may obscure trends, and hide outliers.
#      - **KDEs** can mislead with small data, are sensitive to bandwidth, and don’t show explicit counts.
# 
# Overall, we compared these different data visualization techniques in terms of both their strengths and weaknesses, depending on the kind of insights you want from the data.

# # Question 5: 
# 
# The relationship between Mean, Median, Left Skewness and Right Skewness is that if the median is greater than the mean then the data is left skewed. The inverse relationship where mean is greater than the median then the data is skewed right. Finally if the median and mean are equal then the data would be symmetrical aka no skew. 

# In[23]:


from scipy import stats
import pandas as pd
import numpy as np
  
sample1 = stats.gamma(a=2,scale=2).rvs(size=1000)
fig1 = px.histogram(pd.DataFrame({'data': sample1}), x="data")
# USE `fig1.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

mean1 = sample1.mean()
median1 = np.median(sample1)
np.quantile(sample1, [0.5]) # median

sample2 = -stats.gamma(a=2,scale=2).rvs(size=1000)
mean2 = sample2.mean()
median2 = np.median(sample2)


# In[24]:


fig1.add_vline(x=mean1, line_dash="dash", line_color="blue", annotation_text="Mean", annotation_position="top right")
fig1.add_vline(x=median1, line_dash="dash", line_color="green", annotation_text="Median", annotation_position="top left")
fig1.show(renderer = 'png')


# In[20]:


fig2 = px.histogram(pd.DataFrame({'data': sample2}), x="data")
fig2.add_vline(x=mean2, line_dash="dash", line_color="blue", annotation_text="Mean", annotation_position="top right")
fig2.add_vline(x=median2, line_dash="dash", line_color="green", annotation_text="Median", annotation_position="top left")
fig2.show(renderer = 'png')


# # Question 6:
# 
# I was able to get chat to show me some an interesting dataset about wine quality. I then tried to show the mean and median for different subsets of the wine quality, such as the quality and the amount of alochol that was found. The quality had a more symmetrical representation but the alcohol had a more left skew. 

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')  # The data is separated by semicolons

# Select the 'quality' feature
quality_data = data['quality']

# Calculate mean and median
mean = np.mean(quality_data)
median = np.median(quality_data)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(quality_data, bins=np.arange(2.5, 9.5, 0.5), color='skyblue', alpha=0.7, edgecolor='black')

# Add mean and median lines
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')

# Add titles and labels
plt.title('Wine Quality Histogram with Mean and Median')
plt.xlabel('Quality Ratings')
plt.ylabel('Frequency')
plt.xticks(np.arange(3, 9, 1))
plt.legend()

# Show the plot
plt.show()


# In[27]:


import pandas as pd

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Calculate skewness for each numerical column
skewness = data.skew()

# Display the skewness values
print("Skewness of each column:")
print(skewness)


# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')  # The data is separated by semicolons

# Select the 'quality' feature
quality_data = data['alcohol']

# Calculate mean and median
mean = np.mean(quality_data)
median = np.median(quality_data)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(quality_data, bins=np.arange(2.5, 9.5, 0.5), color='skyblue', alpha=0.7, edgecolor='black')

# Add mean and median lines
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')

# Add titles and labels
plt.title('Wine Quality Histogram with Mean and Median')
plt.xlabel('Quality Ratings')
plt.ylabel('Frequency')
plt.xticks(np.arange(3, 9, 1))
plt.legend()

# Show the plot
plt.show()


# # Question 7: 
# 
# 

# In[41]:


import plotly.express as px
df = px.data.gapminder()
fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
fig.show(renderer = 'png')


# # Question 8: 

# In[ ]:





# In[44]:


bn = pd.read_csv('https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv')
bn['name'] = bn['name']+" "+bn['sex'] # make identical boy and girl names distinct
bn['rank'] = bn.groupby('year')['percent'].rank(ascending=False)
bn = bn.sort_values(['name','year'])
# the next three lines create the increaes or decrease in name prevalence from the last year 
bn['percent change'] = bn['percent'].diff()
new_name = [True]+list(bn.name[:-1].values!=bn.name[1:].values)
bn.loc[new_name,'percentage change'] = bn.loc[new_name,'percent'] 
bn = bn.sort_values('year')
bn = bn[bn.percent>0.001] # restrict to "common" names
fig2 = px.scatter(bn, x="percent change", y="rank", animation_frame="year", animation_group="name",
           size="percent", color="sex", hover_name="name",
           size_max=50, range_x=[-0.005, 0.005])
fig2.update_yaxes(autorange='reversed') # this lets us put rank 1 on the top
fig2.show(renderer = 'png')


# # Question 9:
# Yes
