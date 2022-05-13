#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries 

# In[137]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3


# In[138]:


con = sqlite3.connect('airbnb.db')


# #  Importing the data from the database

# In[139]:


###df_listings=pd.read_sql_table('listings','sqlite:///C:/AirBnb_CaseStudy/AirBnb_CaseStudy.db')

##Another way

df_listings = pd.read_sql_query("SELECT * from listings", con)
df_calender= pd.read_sql_query("SELECT * from Calendar", con)
df_hosts=pd.read_sql_query("SELECT * from Hosts", con)
df_reviews=pd.read_sql_query("SELECT * from Reviews", con)


# # Look at the table Calendar how many rows and unique listing ids are present? Are there any implications when it comes to having more rows and less unique listing ids?
# 
# Look at the price column in Calendar table. What transformations you will need to perform so that you can create a column that can be used as a target/response variable?
# 
# Look at the tables Listings, Hosts and Reviews to come up with a list of potential transformations needed in order to have predictors that can be used to predict the listing price.
# 
# Create an aggregated view of data spread across different tables, containing the target as well as predictor variables.

# # Data header and finding Missing Values

# In[140]:


df_listings.head()


# In[141]:


#missing values in listings table
df_listings.isna().sum().plot(kind='bar')


# In[142]:


df_listings.shape


# In[223]:


df_listings.nunique()


# In[143]:


df_calender.head()


# In[144]:


#missing values in calender  table
df_calender.isna().sum().plot(kind='bar')


# In[145]:


df_calender.shape


# In[146]:


df_calender['listing_id'].nunique()


# In[ ]:





# In[147]:


df_hosts.head()


# In[148]:


#missing values in hosts  table
df_hosts.isna().sum().plot(kind='bar')


# In[149]:


df_hosts.shape


# In[150]:


df_reviews.head()


# In[151]:


#missing values in reviews table
df_reviews.isna().sum().plot(kind='bar')


# In[152]:


df_reviews.shape


# # df_calender and  df_reviews  are not one to one , so we have to make it one to one .
#  

# # Calender Table
# ### Simple Approach Group by func
# 
# 

# In[153]:


# Taking average of all values 


# In[154]:


df_calender.groupby('listing_id').mean()


# In[155]:


## Last Price Approach (take the last price of each listing id )
df_calender.sort_values('date').groupby('listing_id').tail(1)


# In[156]:


df_calender_lst_2=df_calender.sort_values('date').groupby('listing_id').tail(2)
df_calender_lst_2


# In[157]:


df_calender_lst_2.reset_index(level=0, inplace=True)


# In[158]:


df_calender_lst_2.drop(['index'],axis=1)


# In[159]:


#convert to date time 
df_calender_lst_2['date'] = pd.to_datetime(df_calender_lst_2['date'])


# In[160]:


#remove time from stamp
df_calender_lst_2['date'] = df_calender_lst_2['date'].dt.date


# In[161]:


df_calender_lst_2.head()


# In[162]:


df_calender_lst_2.dtypes


# In[ ]:





# # Hosts Table 

# In[163]:


df_hosts.head()


# In[164]:


df_hosts.describe()


# In[165]:


df_hosts.isna().sum().plot(kind='bar')


# In[166]:


df_hosts.shape


# In[167]:


df_hosts.nunique()


# # from the above we can clearly see that host id is unique

# In[168]:


# to convert host_since to time stamp
df_hosts['host_since'] = pd.to_datetime(df_hosts['host_since'],errors='coerce')


# In[169]:


df_hosts['host_since'] = df_hosts['host_since'].dt.date


# In[170]:


df_hosts.head()


# # Reviews Table

# In[171]:


df_reviews.head()


# In[172]:


df_reviews.shape


# In[173]:


df_reviews.isna().sum().plot(kind='bar')


# In[174]:


df_reviews.groupby('listing_id').count()['review_id']


# # for a single listing_id there are multiple reviews

# In[175]:


df_reviews.groupby('reviewer_id').count()['review_id'].sort_values(ascending=False)[:20].plot(kind='bar')


# In[176]:


df_reviews.groupby('reviewer_id').count()['review_id'].sort_values(ascending=True)[:20].plot(kind='bar')


# In[177]:


df_reviews.groupby('listing_id').count()['review_id'].describe()


# In[178]:


df_reviews.groupby('reviewer_id').count()['review_id'].describe()


# In[179]:


df_reviews.groupby('listing_id').count()['review_id'].sort_values(ascending=False)[:20].plot(kind='bar')


# In[180]:


df_reviews.groupby('listing_id').count()['review_id'].sort_values(ascending=True)[:20].plot(kind='bar')


# In[181]:


sns.kdeplot(df_reviews.groupby('listing_id').count()['review_id'])


# In[182]:


sns.kdeplot(df_reviews.groupby('reviewer_id').count()['review_id'])


# ##Days since review and no of review per listing id we are adding two more columns 

# In[183]:


review_per_listingid=df_reviews.groupby('listing_id').count()['review_id'].to_dict()


# In[184]:


df_reviews['review_per_listingid']=df_reviews['listing_id'].map(review_per_listingid)


# In[185]:


df_reviews


# In[186]:


df_reviews['date'] = pd.to_datetime(df_reviews['date'],errors='coerce')


# In[187]:


df_reviews['date'] = df_reviews['date'].dt.date


# In[188]:


df_reviews.dtypes


# In[189]:



#ty_date=pd.Timestamp('2022-05-10')
#no_of_days_since_review=abs(df_reviews['date'] - ty_date.days)
#no_of_days_since_review


# In[190]:


ty_date=pd.to_datetime("today").strftime("%m/%d/%Y")


# In[191]:


#t_dat= pd.DataFrame(data=ty_date)
#t_dat


# In[192]:


df_reviews['date']=df_reviews['date'].astype(str)
df_reviews.dtypes


# In[193]:


#df_ty_date= pd.DataFrame()
#df_reviews["col1"] = ty_date


# In[194]:


df_reviews['todays date'] = pd.to_datetime('today')
df_reviews.head()


# In[195]:



df_reviews['todays date'] = pd.to_datetime(df_reviews['todays date'],errors='coerce')


# In[196]:


df_reviews['todays date'] = df_reviews['todays date'].dt.date


# In[197]:


df_reviews.head()


# In[217]:


#df_reviews.dtypes
#df_reviews['todays date'] =df_reviews['todays date'] .strftime('%m/%d/%Y')
#df_reviews['todays date'] 

df_reviews['todays date']=df_reviews['todays date'].to_frame()
df_reviews['todays date'].reset_index()
df_reviews


# In[220]:


df_reviews['dates'] =df_reviews['date']
df_reviews


# In[222]:


df_reviews['dates'].to_datetime()


# In[209]:


birthdate = datetime.datetime.strptime(df_reviews['date'],'%m/%d/%Y')
currentDate = datetime.datetime.today()

days = birthdate - currentDate


# In[ ]:




