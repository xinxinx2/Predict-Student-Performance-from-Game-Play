# Random Forest Baseline - LB 0.664
In this notebook we present a Random Forest baseline. We train GroupKFold models for each of the 18 questions. Our CV score is 0.664. We infer test using one of our KFold models. We can improve our CV and LB by engineering more features for our random forest and/or trying different models (like other ML models and/or RNN and/or Transformer). Also we can improve our LB by using more KFold models OR training one model using all data (and the hyperparameters that we found from our KFold cross validation).


```python
import pandas as pd, numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import gc
import random
from tqdm import tqdm
```

# Load Train Data and Labels


```python
f_label=open('/kaggle/input/predict-student-performance-from-game-play/train_labels.csv')
for i,line in enumerate(f_label):
    pass
print(i)
```

    424116



```python
# 采样一部分行
def sample_row():
    with open('/kaggle/input/predict-student-performance-from-game-play/train.csv') as f:
#         f_label=open('/kaggle/input/predict-student-performance-from-game-play/train_labels.csv')
        L=26296947
        head=f.readline()
#         head_label=f_label.readline()
        print(head.strip())
        with open('/kaggle/working/train_sample_rows.csv','w') as fw:
#             label_fw=open('/kaggle/working/train_label_sample_rows.csv','w')
            fw.write(head)
#             label_fw.write(head_label)
            for line in tqdm(f,total=L):
#                 line_label=f_label.readline()
                if random.random()<1/5000:
                    fw.write(line)
#                     label_fw.write(line_label)
#         f_label.close()
#         label_fw.close()
sample_row()
gc.collect()
```

    session_id,index,elapsed_time,event_name,name,level,page,room_coor_x,room_coor_y,screen_coor_x,screen_coor_y,hover_duration,text,fqid,room_fqid,text_fqid,fullscreen,hq,music,level_group


    100%|█████████▉| 26296946/26296947 [00:59<00:00, 440252.50it/s]





    0




```python
# 采样一部分列
# 26296947
def sample_column():
    with open('/kaggle/input/predict-student-performance-from-game-play/train.csv') as f:
        head=f.readline()
    #     print(head)
        head=head.strip().split(',')
        remove_i=[head.index('text'),head.index('room_fqid'),head.index('text_fqid')]
    #     remove_i=remove_i[::-1]
        remove_i.sort(reverse=True)
        print(remove_i)
        for i in remove_i:
            head.pop(i)
        head=','.join(head)+'\n'
    #     print(i,head.strip().split(',')[i])
        with open('/kaggle/working/train_sample_columns.csv','w') as fw:
            fw.write(head)
            for line in tqdm(f,total=26296946):
                line=line.strip().split(',')
                for i in remove_i:
                    line.pop(i)
                line=','.join(line)+'\n'
                fw.write(line)
# sample_column()
```


```python
train = pd.read_csv('/kaggle/working/train_sample_rows.csv',memory_map=True)
print( train.shape )
train.head()
```

    (5118, 20)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session_id</th>
      <th>index</th>
      <th>elapsed_time</th>
      <th>event_name</th>
      <th>name</th>
      <th>level</th>
      <th>page</th>
      <th>room_coor_x</th>
      <th>room_coor_y</th>
      <th>screen_coor_x</th>
      <th>screen_coor_y</th>
      <th>hover_duration</th>
      <th>text</th>
      <th>fqid</th>
      <th>room_fqid</th>
      <th>text_fqid</th>
      <th>fullscreen</th>
      <th>hq</th>
      <th>music</th>
      <th>level_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20090312431273200</td>
      <td>434</td>
      <td>463665</td>
      <td>person_click</td>
      <td>basic</td>
      <td>11</td>
      <td>NaN</td>
      <td>-0.348501</td>
      <td>-167.000000</td>
      <td>408.0</td>
      <td>497.0</td>
      <td>NaN</td>
      <td>Wait a sec. Women couldn't vote?!</td>
      <td>archivist</td>
      <td>tunic.historicalsociety.frontdesk</td>
      <td>tunic.historicalsociety.frontdesk.archivist.ne...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5-12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20090312455206810</td>
      <td>86</td>
      <td>254638</td>
      <td>navigate_click</td>
      <td>undefined</td>
      <td>2</td>
      <td>NaN</td>
      <td>-966.951436</td>
      <td>-64.444915</td>
      <td>32.0</td>
      <td>495.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>tocollection</td>
      <td>tunic.historicalsociety.entry</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0-4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20090313091715820</td>
      <td>419</td>
      <td>614783</td>
      <td>person_click</td>
      <td>basic</td>
      <td>10</td>
      <td>NaN</td>
      <td>-230.687524</td>
      <td>158.724424</td>
      <td>421.0</td>
      <td>242.0</td>
      <td>NaN</td>
      <td>What was Wells doing here?</td>
      <td>worker</td>
      <td>tunic.library.frontdesk</td>
      <td>tunic.library.frontdesk.worker.wells</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5-12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20090313091715820</td>
      <td>581</td>
      <td>1002320</td>
      <td>navigate_click</td>
      <td>undefined</td>
      <td>13</td>
      <td>NaN</td>
      <td>273.952706</td>
      <td>-152.974858</td>
      <td>981.0</td>
      <td>479.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>tunic.historicalsociety.entry</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13-22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20090315085850788</td>
      <td>549</td>
      <td>1559721</td>
      <td>navigate_click</td>
      <td>undefined</td>
      <td>13</td>
      <td>NaN</td>
      <td>151.384955</td>
      <td>-51.673082</td>
      <td>782.0</td>
      <td>589.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>tunic.historicalsociety.basement</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>13-22</td>
    </tr>
  </tbody>
</table>
</div>




```python
targets = pd.read_csv('/kaggle/input/predict-student-performance-from-game-play/train_labels.csv')
targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]) )
targets['q'] = targets.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )
print( targets.shape )
targets.head()
```

    (424116, 4)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session_id</th>
      <th>correct</th>
      <th>session</th>
      <th>q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20090312431273200_q1</td>
      <td>1</td>
      <td>20090312431273200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20090312433251036_q1</td>
      <td>0</td>
      <td>20090312433251036</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20090312455206810_q1</td>
      <td>1</td>
      <td>20090312455206810</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20090313091715820_q1</td>
      <td>0</td>
      <td>20090313091715820</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20090313571836404_q1</td>
      <td>1</td>
      <td>20090313571836404</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
gc.collect()
```




    48



# Feature Engineer
We create basic aggregate features. Try creating more features to boost CV and LB!


```python
CATS = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid']
NUMS = ['elapsed_time','level','page','room_coor_x', 'room_coor_y', 
        'screen_coor_x', 'screen_coor_y', 'hover_duration']
```


```python
def feature_engineer(train):
    dfs = []
    for c in CATS:
        tmp = train.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in NUMS:
        tmp = train.groupby(['session_id','level_group'])[c].agg('mean')
        dfs.append(tmp)
    for c in NUMS:
        tmp = train.groupby(['session_id','level_group'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        dfs.append(tmp)
    df = pd.concat(dfs,axis=1)
    df = df.fillna(-1)
    df = df.reset_index()
    df = df.set_index('session_id')
    return df
```


```python
%%time
df = feature_engineer(train)
print( df.shape )
df.head()
```

    (4892, 22)
    CPU times: user 81 ms, sys: 959 µs, total: 82 ms
    Wall time: 88.6 ms





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>level_group</th>
      <th>event_name_nunique</th>
      <th>name_nunique</th>
      <th>fqid_nunique</th>
      <th>room_fqid_nunique</th>
      <th>text_fqid_nunique</th>
      <th>elapsed_time</th>
      <th>level</th>
      <th>page</th>
      <th>room_coor_x</th>
      <th>...</th>
      <th>screen_coor_y</th>
      <th>hover_duration</th>
      <th>elapsed_time_std</th>
      <th>level_std</th>
      <th>page_std</th>
      <th>room_coor_x_std</th>
      <th>room_coor_y_std</th>
      <th>screen_coor_x_std</th>
      <th>screen_coor_y_std</th>
      <th>hover_duration_std</th>
    </tr>
    <tr>
      <th>session_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20090312431273200</th>
      <td>5-12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>463665.0</td>
      <td>11.0</td>
      <td>-1.0</td>
      <td>-0.348501</td>
      <td>...</td>
      <td>497.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>20090312455206810</th>
      <td>0-4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>254638.0</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>-966.951436</td>
      <td>...</td>
      <td>495.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>20090313091715820</th>
      <td>13-22</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1002320.0</td>
      <td>13.0</td>
      <td>-1.0</td>
      <td>273.952706</td>
      <td>...</td>
      <td>479.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>20090313091715820</th>
      <td>5-12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>614783.0</td>
      <td>10.0</td>
      <td>-1.0</td>
      <td>-230.687524</td>
      <td>...</td>
      <td>242.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>20090315085850788</th>
      <td>13-22</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1559721.0</td>
      <td>13.0</td>
      <td>-1.0</td>
      <td>151.384955</td>
      <td>...</td>
      <td>589.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



# Train Random Forest Model
We train one model for each of 18 questions. Furthermore, we use data from `level_groups = '0-4'` to train model for questions 1-3, and `level groups '5-12'` to train questions 4 thru 13 and `level groups '13-22'` to train questions 14 thru 18. Because this is the data we get (to predict corresponding questions) from Kaggle's inference API during test inference. We can improve our model by saving a user's previous data from earlier `level_groups` and using that to predict future `level_groups`.


```python
FEATURES = [c for c in df.columns if c != 'level_group']
print('We will train with', len(FEATURES) ,'features')
ALL_USERS = df.index.unique()
print('We will train with', len(ALL_USERS) ,'users info')
```

    We will train with 21 features
    We will train with 4578 users info



```python
targets.shape
```




    (424116, 4)




```python
gkf = GroupKFold(n_splits=5)
oof = pd.DataFrame(data=np.zeros((len(ALL_USERS),18)), index=ALL_USERS)
models = {}

# COMPUTE CV SCORE WITH 5 GROUP K FOLD
for i, (train_index, test_index) in enumerate(gkf.split(X=df, groups=df.index)):
    print('#'*25)
    print('### Fold',i+1)
    print('#'*25)
    
    # ITERATE THRU QUESTIONS 1 THRU 18
    for t in range(1,19):
        print(t,', ',end='')
        
        # USE THIS TRAIN DATA WITH THESE QUESTIONS
        if t<=3: grp = '0-4'
        elif t<=13: grp = '5-12'
        elif t<=22: grp = '13-22'
            
        # TRAIN DATA
        train_x = df.iloc[train_index]
        train_x = train_x.loc[train_x.level_group == grp]
        train_users = train_x.index.values
        train_y = targets.loc[targets.q==t].set_index('session').loc[train_users]
        
        # VALID DATA
        valid_x = df.iloc[test_index]
        valid_x = valid_x.loc[valid_x.level_group == grp]
        valid_users = valid_x.index.values
        valid_y = targets.loc[targets.q==t].set_index('session').loc[valid_users]
        
        # TRAIN MODEL
        clf = RandomForestClassifier() 
        clf.fit(train_x[FEATURES].astype('float32'), train_y['correct'])
        
        # SAVE MODEL, PREDICT VALID OOF
        models[f'{grp}_{t}'] = clf
        oof.loc[valid_users, t-1] = clf.predict_proba(valid_x[FEATURES].astype('float32'))[:,1]
        
    print()
```

    #########################
    ### Fold 1
    #########################
    1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 
    #########################
    ### Fold 2
    #########################
    1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 
    #########################
    ### Fold 3
    #########################
    1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 
    #########################
    ### Fold 4
    #########################
    1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 
    #########################
    ### Fold 5
    #########################
    1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 


# Compute CV Score
We need to convert prediction probabilities into `1s` and `0s`. The competition metric is F1 Score which is the harmonic mean of precision and recall. Let's find the optimal threshold for `p > threshold` when to predict `1` and when to predict `0` to maximize F1 Score.


```python
# PUT TRUE LABELS INTO DATAFRAME WITH 18 COLUMNS
true = oof.copy()
for k in range(18):
    # GET TRUE LABELS
    tmp = targets.loc[targets.q == k+1].set_index('session').loc[ALL_USERS]
    true[k] = tmp.correct.values
```


```python
# FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
scores = []; thresholds = []
best_score = 0; best_threshold = 0

for threshold in np.arange(0.4,0.81,0.01):
    print(f'{threshold:.02f}, ',end='')
    preds = (oof.values.reshape((-1))>threshold).astype('int')
    m = f1_score(true.values.reshape((-1)), preds, average='macro')   
    scores.append(m)
    thresholds.append(threshold)
    if m>best_score:
        best_score = m
        best_threshold = threshold
```

    0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 


```python
import matplotlib.pyplot as plt

# PLOT THRESHOLD VS. F1_SCORE
plt.figure(figsize=(20,5))
plt.plot(thresholds,scores,'-o',color='blue')
plt.scatter([best_threshold], [best_score], color='blue', s=300, alpha=1)
plt.xlabel('Threshold',size=14)
plt.ylabel('Validation F1 Score',size=14)
plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3}',size=18)
plt.show()
```


```python
print('When using optimal threshold...')
for k in range(18):
        
    # COMPUTE F1 SCORE PER QUESTION
    m = f1_score(true[k].values, (oof[k].values>best_threshold).astype('int'), average='macro')
    print(f'Q{k}: F1 =',m)
    
# COMPUTE F1 SCORE OVERALL
m = f1_score(true.values.reshape((-1)), (oof.values.reshape((-1))>best_threshold).astype('int'), average='macro')
print('==> Overall F1 =',m)
```

    When using optimal threshold...
    Q0: F1 = 0.34891327749602924
    Q1: F1 = 0.17116937684164577
    Q2: F1 = 0.20930101694010494
    Q3: F1 = 0.41078816425795994
    Q4: F1 = 0.47484485319957825
    Q5: F1 = 0.4230817959475843
    Q6: F1 = 0.4447469021011702
    Q7: F1 = 0.4548393285918271
    Q8: F1 = 0.4422010881191989
    Q9: F1 = 0.4769569061415597
    Q10: F1 = 0.4544571342802953
    Q11: F1 = 0.3745727755274061
    Q12: F1 = 0.4802580594207978
    Q13: F1 = 0.4659641054992407
    Q14: F1 = 0.4858242703917467
    Q15: F1 = 0.4766656612164618
    Q16: F1 = 0.4861475589856278
    Q17: F1 = 0.39853043389474385
    ==> Overall F1 = 0.4507073074704827


# Infer Test Data


```python
import jo_wilder
env = jo_wilder.make_env()
iter_test = env.iter_test()
```


```python
tmp=[]
i=0
for (test, sample_submission) in iter_test:
    print(i)
    i+=1
    x = feature_engineer(test)
    y = sample_submission.copy()
    y['session'] = y.session_id.apply(lambda x: int(x.split('_')[0]) )
    y['q'] = y.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )
    y=y.set_index('session_id')
    for t in range(1,19):
        if t<=3: grp = '0-4'
        elif t<=13: grp = '5-12'
        elif t<=22: grp = '13-22'
        x_ = x.loc[x.level_group == grp]
        users = x_.index.values
        clf = models[f'{grp}_{t}']
        p = clf.predict_proba(x_[FEATURES].astype('float32'))[:,1]
        y.loc[x_.index.map(lambda x:str(x)+f'_q{t}'),'correct']=p>best_threshold
    y.correct=y.correct.astype('int')
    sample_submission['correct']=y.loc[sample_submission.session_id,'correct'].values
    tmp.append(sample_submission)
    env.predict(sample_submission)
```

    This version of the API is not optimized and should not be used to estimate the runtime of your code on the hidden test set.
    0



```python
sample_submission
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session_id</th>
      <th>correct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20090109393214576_q1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20090312143683264_q1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20090312331414616_q1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20090109393214576_q2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20090312143683264_q2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20090312331414616_q2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20090109393214576_q3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20090312143683264_q3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20090312331414616_q3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20090109393214576_q4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>20090312143683264_q4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>20090312331414616_q4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>20090109393214576_q5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20090312143683264_q5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>20090312331414616_q5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>20090109393214576_q6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>20090312143683264_q6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>20090312331414616_q6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>20090109393214576_q7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20090312143683264_q7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20090312331414616_q7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>20090109393214576_q8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>20090312143683264_q8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>20090312331414616_q8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>20090109393214576_q9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>20090312143683264_q9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>20090312331414616_q9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>20090109393214576_q10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>20090312143683264_q10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>20090312331414616_q10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>20090109393214576_q11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>20090312143683264_q11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>20090312331414616_q11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>20090109393214576_q12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>20090312143683264_q12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>20090312331414616_q12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>20090109393214576_q13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>20090312143683264_q13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>20090312331414616_q13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>20090109393214576_q14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40</th>
      <td>20090312143683264_q14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>20090312331414616_q14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42</th>
      <td>20090109393214576_q15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>20090312143683264_q15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>20090312331414616_q15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>20090109393214576_q16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>20090312143683264_q16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>20090312331414616_q16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>20090109393214576_q17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>20090312143683264_q17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>20090312331414616_q17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51</th>
      <td>20090109393214576_q18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52</th>
      <td>20090312143683264_q18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53</th>
      <td>20090312331414616_q18</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# EDA submission.csv


```python
df = pd.read_csv('submission.csv')
print( df.shape )
df.head()
```


```python
print(df.correct.mean())
```
