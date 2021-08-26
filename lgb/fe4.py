#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


for i in range(0,16):
    print(i)
    day = 116 - i*7


# In[3]:


    train = pd.read_csv('../input/train.csv')
    
    
    # In[4]:
    
    
    train = train.sort_values(by='article_id')
    
    
    # In[5]:
    
    
    if day<=109:
        valid = train[(train.date>day)&(train.date<=(day+7))]
    else:
        valid = pd.read_csv('../input/test.csv')
        valid = valid.sort_values(by='article_id')
    
    
    # In[6]:
    
    
    train = train[train.date<=day]
    
    
    # In[7]:
    
    
    valid['date_count'] = valid['date'].map(valid['date'].value_counts())
    
    
    # In[8]:
    
    
    valid['date_count'] = valid.groupby('date').cumcount()/valid['date_count']
    
    
    # In[9]:
    
    
    valid['date_7'] = valid['date']%7 
    
    
    # In[10]:
    
    
    for column in ['price', 'price_diff']:
        valid[column+'_int'] = valid[column].fillna(-1).astype('int')
        valid[column+'_int_last'] = valid[column+'_int']%10 #last int 
        valid[column+'_decimal'] = round(((valid[column]-valid[column+'_int'])*100).fillna(-1)).astype('int')    #decimal
        valid[column+'_decimal_is_0'] = (valid[column+'_decimal']==0).astype('int')
        valid[column+'_decimal_is_5'] = (valid[column+'_decimal']%5==0).astype('int') 
        valid[column+'_decimal_is_9'] = (valid[column+'_decimal']%9==0).astype('int') 
        valid[column+'_decimal_last'] = valid[column+'_decimal']%10
        valid[column+'_decimal_last2'] = valid[column+'_decimal']//5 
    
    
    # In[11]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_mean_orders_3h_15h'] = valid[col].map(train.groupby(col)['orders_3h_15h'].mean())
    
    
    # In[12]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_count_now'] = valid[col].map(valid.groupby(col).size())
    
    
    # In[13]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_count'] = valid[col].map(train[train.date>(day-28)].groupby(col).size())
    
    
    # In[14]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_mean_orders_3h_15h_last_28'] = valid[col].map(train[train.date>(day-28)].groupby(col)['orders_3h_15h'].mean())
    
    
    # In[15]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_mean_orders_2h_now'] = valid[col].map(valid.groupby(col)['orders_2h'].mean())
    
    
    # In[16]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_mean_price_now'] = valid[col].map(valid.groupby(col)['price'].mean())
    
    
    # In[17]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_mean_price'] = valid[col].map(train.groupby(col)['price'].mean())
    
    
    # In[18]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        tmp = valid.groupby([col,'date'])['orders_2h'].mean().reset_index().rename(columns = {'orders_2h' : col+'_mean_orders_2h_now2'})
        valid = valid.merge(tmp,on=[col,'date'],how='left')
    
    
    # In[19]:
    
    
    valid['orders_2h_favorite'] = valid['orders_2h'] / (valid['favorite_2h']+0.0001)
    
    
    # In[20]:
    
    
    for col in [ 'baike_id_1h',  'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        tmp =  train.groupby(['author',col])['orders_3h_15h'].mean().reset_index().rename(columns = {'orders_3h_15h' :'author_'+col+'_mean_orders_3h_15h'})
        valid = valid.merge(tmp,on=['author',col],how='left')
    
    
    # In[21]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_mean_favorite_2h'] = valid[col].map(train.groupby(col)['favorite_2h'].mean())
    
    
    # In[22]:
    
    
    valid['date_count_1'] = valid['date_count'] - valid.groupby(['url','date'])['date_count'].shift(-1)
    valid['date_count_2'] = valid['date_count'] - valid.groupby(['url','date'])['date_count'].shift(1)
    
    
    # In[23]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_mean_orders_zengzhang'] = valid[col].map(train.groupby(col)['orders_2h'].mean())/valid[col+'_mean_orders_3h_15h']
    
    
    # In[24]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_mean_orders_3h_15h_orders_2h_fei0'] = valid[col].map(train[train.orders_2h>0].groupby(col)['orders_3h_15h'].mean())
    
    
    # In[25]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_mean_orders_3h_15h_orders_2h_0'] = valid[col].map(train[train.orders_2h==0].groupby(col)['orders_3h_15h'].mean())
    
    
    # In[26]:
    
    
    for col in [ 'baike_id_1h', 'author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall', 'url','baike_id_2h' ]:
        valid[col+'_mean_orders_3h_15h_last_7'] = valid[col].map(train[train.date>(day-7)].groupby(col)['orders_3h_15h'].mean())
    
    
    # In[ ]:
    
    
    
    
    
    # In[27]:
    
    
    valid.reset_index(drop=True).to_feather('../fe/fe_{}.feather'.format(day))
    

# In[ ]:




