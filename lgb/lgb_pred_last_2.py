#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils import shuffle
cache_path = '../cache/'
seed = 2020
fe_path = '../fe/'
from collections import defaultdict





train = pd.read_feather(fe_path+'fe_{}.feather'.format(109))
valid_shape = train.shape[0]
for i in range(1,10):
    train_tmp = pd.read_feather(fe_path+'fe_{}.feather'.format(109-i*7))
    train = train.append(train_tmp).reset_index(drop=True)

test = pd.read_feather(fe_path+'fe_{}.feather'.format(116))
train = train.append(test).reset_index(drop=True)



#train['prior_question_had_explanation'] = train.prior_question_had_explanation.astype('float')
###################################################################################################################


def kfold_lightgbm(params,df, predictors,target,num_folds, stratified = True,
                  objective='mse', metrics='mse',debug= False,
                   feval=None, early_stopping_rounds=100, num_boost_round=1000, verbose_eval=50, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
    }

    lgb_params.update(params)
    
    train_df = df[df[target].notnull()]
    test_df = df[df[target].isnull()]   
    
    print("Starting LightGBM. Train shape: {}".format(train_df.shape))
    gc.collect()
 

    train_x, train_y = train_df[feats], train_df[target]
    print(train_x.shape)


    xgtrain = lgb.Dataset(train_x.values, label=train_y.values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
#    xgvalid = lgb.Dataset(valid_x.values, label=valid_y.values,
#                          feature_name=predictors,
#                          categorical_feature=categorical_features
#                          )


    clf = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, ], 
                     valid_names=['train'], 
                     num_boost_round=num_boost_round,
#                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=verbose_eval, 
                     feval=feval)



    
    sub_preds = clf.predict(test_df[feats], num_iteration=num_boost_round)
    sub_preds =  np.clip(sub_preds,0,np.inf)
    test_df[target] = sub_preds
    test_df[[ID, target]].to_csv('../res/last_2.csv', index= False)



def display_importances(feature_importance_df_,score):
    ft = feature_importance_df_[["feature", "split","gain"]].groupby("feature").mean().sort_values(by="split", ascending=False)
    print(ft.head(60))
    ft.to_csv('../tiaotz/'+'importance_lightgbm_{}.csv'.format(score),index=True)
    cols = ft[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="split", y="feature", data=best_features.sort_values(by="split", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
#    plt.savefig('lgbm_importances_{}.png'.format(score))
    
    
###################################################################################################################

    
params = {'num_leaves': 128,
         'min_data_in_leaf': 20,
         'learning_rate': 0.01,
         'max_depth': 7,
         "boosting": "gbdt",
         "feature_fraction": 0.7,
         "bagging_freq": 1,
         "bagging_fraction": 0.7,
         "bagging_seed": 42,
         "lambda_l1":1.15,
         "lambda_l2":4.632,
         "verbosity": -1,
         "random_state": 42}



    

no_use= ['date',
 'date_count_2',
 'level4_mean_orders_3h_15h_orders_2h_fei0',
 'author_mean_orders_zengzhang',
 'level4_mean_favorite_2h',
 'level3_mean_orders_zengzhang',
 'author_mean_price_now',
 'level4',
 'baike_id_2h_mean_orders_zengzhang',
 'price_int_last',
 'brand_mean_orders_zengzhang',
 'level4_mean_orders_zengzhang',
 'level2_mean_orders_3h_15h_orders_2h_fei0',
 'level3_mean_favorite_2h',
 'price_diff_decimal',
 'level3_mean_orders_3h_15h_orders_2h_fei0',
 'level4_mean_price_now',
 'level3_mean_price_now',
 'level2_mean_orders_3h_15h_orders_2h_0',
 'level4_mean_orders_2h_now2',
 'level4_count',
 'level2_mean_orders_3h_15h_last_28',
 'level3_mean_orders_3h_15h_last_28',
 'url_mean_price',
 'author_mean_price',
 'brand_mean_price_now',
 'baike_id_1h_mean_orders_zengzhang',
 'level2_mean_orders_3h_15h',
 'brand',
 'baike_id_2h_mean_favorite_2h',
 'baike_id_1h_mean_favorite_2h',
 'level3_mean_price',
 'level4_mean_price',
 'level1_mean_orders_3h_15h_last_28']





ID = 'article_id'



target = 'orders_3h_15h'
        


no_use2 = [ ]

no_use_col = [target]+no_use+no_use2+[ID]
feats = [f for f in train.columns if f not in no_use_col]


categorical_columns = [col for col in feats if train[col].dtype == 'object']

for feature in categorical_columns:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()    
    train[feature] = encoder.fit_transform(train[feature].astype(str))  


    


categorical_columns = []

#train[feats] = train[feats].fillna(-1)
#train = reduce_mem_usage(train)

clf = kfold_lightgbm(params,train,feats,target,5,num_boost_round=6200,early_stopping_rounds=6200,categorical_features=categorical_columns)



