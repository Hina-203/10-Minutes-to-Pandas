#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Here, we import as follows 
   
import numpy as np
import pandas as pd
   


# In[9]:


# Object Creation

# Creating a Series by passing a list of values, 
# letting pandas create a default RangeIndex.

s= pd.Series([1,2,3,4,5,6,7, 'abc', 7.7])
s


# In[10]:


# Creating a DataFrame by passing NumPy using date_range() and labelled columns

dates =pd.date_range("20240515", periods =7)


# In[12]:


dates


# In[19]:


df =pd.DataFrame(np.random.randn(7,4), index= dates, columns =list("ABCD"))


# In[20]:


df


# In[22]:


#Creating a DataFrame by passing a dictionary of objects 
# where the keys are the column labels and the values are the column values.


# In[27]:


df2 =pd.DataFrame(
    { 
        "A" :1.0,
        "B" : pd.Timestamp("20240515"),
        "C" : pd.Series(1, index =list(range(4)), dtype ="float32" ), 
        "D": np.array([3]*4, dtype = "int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F":"foo",
        
    }
)


# In[28]:


df2


# In[32]:


## The columns of the resulting DataFrame have different dtypes:

df2.dtypes


# Viewing Data using Head and Tail
# 

# In[40]:


#DataFrame.head() and DataFrame.tail()
   
df2.head()


# In[38]:


df2.tail()


# In[43]:


df.head(2)


#  Return a NumPy representation of the underlying data with DataFrame.to_numpy() without the index or column labels:

# In[45]:


df.to_numpy()


# In[50]:


df2.to_numpy()


# NumPy arrays have one dtype for the entire array while pandas DataFrames have one dtype per column

# In[51]:


df3 =df2.to_numpy()
df3


# describe() shows a quick statistic summary of your data:
# 

# In[54]:


df2.describe()


# Transposing Data -T -changes the order of columns/ Data

# In[56]:


df2.T


# DataFrame.sort_index() sorts by an axis:

# In[58]:


df2.sort_index(axis=1, ascending= False)


# DataFrame.sort_values() sorts by values:
# 

# In[63]:


df.sort_values(by="D")


#  # Selection
#     While standard Python / NumPy expressions,
#     we recommend the optimized pandas data access methods, DataFrame.at(), DataFrame.iat(), DataFrame.loc() and DataFrame.iloc().
#  

# Getitem ([])- need to pass a column

# In[66]:


df2["B"]


# For a DataFrame, passing a slice : selects matching rows:

# In[70]:


df2[0:3]


# In[71]:


df["20240515":"20240518"]


# Selection by label
# using DataFrame.loc() or DataFrame.at().

# In[72]:


df.loc[dates[1]]                                                      


# Selecting all rows (:) with a select column labels:

# In[79]:


df2.loc[:,"A":"B"]                              #


# For label slicing, both endpoints are included:

# In[83]:


df.loc["20240515":"20240517",["A", "B"]]


# Selecting a single row and column label returns a scalar:

# In[84]:


df.loc[dates[0], "A"]


# For getting fast access to a scalar (equivalent to the prior method):

# In[85]:


df.at[dates[0], "A"]


# # Selection by position
# See more in Selection by Position using DataFrame.iloc() or DataFrame.iat().

# In[88]:


df.iloc[3]


# Integer slices acts similar to NumPy/Python:

# In[93]:


df.iloc[[1,2,4],[0,2]]


# For slicing rows explicitly:

# In[94]:


df.iloc[1:3, :]


# For slicing columns explicitly:

# In[95]:


df.iloc[:, 1:3]


# For getting a value explicitly:

# In[96]:


df.iloc[1, 1]


# For getting fast access to a scalar (equivalent to the prior method):

# In[97]:


df.iat[1, 1]


# # Boolean indexing
# Select rows where df.A is greater than 0.
# 

# In[100]:


df[df["A"] > 0]


# Selecting values from a DataFrame where a boolean condition is met:

# In[101]:


df[df > 0]


# Using isin() method for filtering:

# In[111]:


df3 = df.copy()


# In[112]:


df3["E"] = ["one", "one", "two", "three", "four", "three", "two"]


# In[113]:


df3


# In[114]:


df3[df3["E"].isin(["two", "four"])]


# # Setting
# Setting a new column automatically aligns the data by the indexes:

# In[121]:


s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20240515", periods=6)


# In[122]:


s1


# In[124]:


df["F"] = s1


# In[129]:


#Setting values by label:

df.at[dates[0], "A"] = 0


# In[130]:


df


# In[128]:


# Setting values by position:

df.iat[0, 1] = 0


# Setting by assigning with a NumPy array:

# In[131]:


df.loc[:, "D"] = np.array([5] * len(df))


# In[132]:


df


# A where operation with setting:

# In[133]:


df2 = df.copy()
df2[df2>0] =-df2


# In[134]:


df2


# # Missing data
# For NumPy data types, np.nan represents missing data.
# Reindexing allows you to change/add/delete the index on a specified axis.
# returns a copy of the data:

# In[136]:


df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])


# In[137]:


df1


# DataFrame.dropna() drops any rows that have missing data:

# In[141]:


df1.dropna()


# In[145]:


df1.dropna(how ="any")


# # DataFrame.fillna() fills missing data:

# In[153]:


df1.fillna(value =3)


# isna() gets the boolean mask where values are nan:

# In[154]:


pd.isna(df1)


# ## Operations
# Stats

# Calculate the mean value for each column:

# In[155]:


df.mean()


# Calculate the mean value for each row:

# In[157]:


df.mean(axis=1)


# In[ ]:





# In[ ]:





# In[ ]:




