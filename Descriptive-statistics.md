## Header

Author : Amina Matt    
Date created : 25.04.2022  
Date last mofidied : 25.04.2022  
Description :  Pre-processing of publicly available [Ukraine Conflict Twitter Dataset](https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows) and descriptive statistics


```python
### Libraries

import pandas as pd
import numpy as np
#import twitter
import zipfile

import nltk #natural language processing library
nltk.download('stopwords') #common english words to ignore 
from bs4 import BeautifulSoup #extraction from HTML and XML files
from collections import Counter #dictionary subclass for counting hashable objects

import swifter # for optimization of apply

# Parameters
pd.set_option('display.max_colwidth', 255)
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/aminamatt/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



    ----------------------------------------------------------------

    ModuleNotFoundError            Traceback (most recent call last)

    <ipython-input-1-00f4a4b475aa> in <module>
         11 from collections import Counter #dictionary subclass for counting hashable objects
         12 
    ---> 13 import swifter # for optimization of apply
         14 
         15 # Parameters


    ModuleNotFoundError: No module named 'swifter'


### Path 


```python
# The dataset is store on an external volume
PATH = '/Volumes/PECHE/archive_Twitter_Conflict.zip'
gen_data_PATH = '/Users/aminamatt/Dropbox/Cours-printemps-2022/Computational-Social-Media/Project/Generated_Data/'
```


```python
#pd.read_csv('/Volumes/PECHE/archive_Twitter_Conflict.zip',compression='zip')
```


```python
all_zips = zipfile.ZipFile(PATH)
all_zips
zipfile.ZipFile(PATH).namelist()
```




    ['0401_UkraineCombinedTweetsDeduped.csv.gzip',
     '0402_UkraineCombinedTweetsDeduped.csv.gzip',
     '0403_UkraineCombinedTweetsDeduped.csv.gzip',
     '0404_UkraineCombinedTweetsDeduped.csv.gzip',
     '0405_UkraineCombinedTweetsDeduped.csv.gzip',
     '0406_UkraineCombinedTweetsDeduped.csv.gzip',
     '0407_UkraineCombinedTweetsDeduped.csv.gzip',
     '0408_UkraineCombinedTweetsDeduped.csv.gzip',
     '0409_UkraineCombinedTweetsDeduped.csv.gzip',
     '0410_UkraineCombinedTweetsDeduped.csv.gzip',
     '0411_UkraineCombinedTweetsDeduped.csv.gzip',
     '0412_UkraineCombinedTweetsDeduped.csv.gzip',
     '0413_UkraineCombinedTweetsDeduped.csv.gzip',
     '0414_UkraineCombinedTweetsDeduped.csv.gzip',
     '0415_UkraineCombinedTweetsDeduped.csv.gzip',
     '0416_UkraineCombinedTweetsDeduped.csv.gzip',
     '0417_UkraineCombinedTweetsDeduped.csv.gzip',
     '0418_UkraineCombinedTweetsDeduped.csv.gzip',
     '0419_UkraineCombinedTweetsDeduped.csv.gzip',
     '0420_UkraineCombinedTweetsDeduped.csv.gzip',
     '0421_UkraineCombinedTweetsDeduped.csv.gzip',
     '0422_UkraineCombinedTweetsDeduped.csv.gzip',
     'UkraineCombinedTweetsDeduped20220227-131611.csv.gzip',
     'UkraineCombinedTweetsDeduped_FEB27.csv.gzip',
     'UkraineCombinedTweetsDeduped_FEB28_part1.csv.gzip',
     'UkraineCombinedTweetsDeduped_FEB28_part2.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR01.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR02.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR03.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR04.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR05.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR06.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR07.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR08.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR09.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR10.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR11.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR12.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR13.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR14.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR15.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR16.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR17.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR18.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR19.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR20.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR21.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR22.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR23.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR24.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR25.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR26.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR27_to_28.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR29.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR30_REAL.csv.gzip',
     'UkraineCombinedTweetsDeduped_MAR31.csv.gzip']



## 1. Loading  data


```python
# # # # the zip archive contains multiple files

# all_zips = zipfile.ZipFile(PATH)

# #df = pd.concat(
# #    [pd.read_csv(zipfile.ZipFile(PATH).open(i)) for i in zipfile.ZipFile(PATH).namelist()],
# #    ignore_index=True
# #)
# df = pd.DataFrame()
# index = 0 
# for i in all_zips.namelist():
#     if (i.startswith('Ukraine')):
#         print(f'Loading {i}')
#         tmp = pd.read_csv(zipfile.ZipFile(PATH).open(i),compression='gzip')  # add options index_col=0,encoding='utf-8', quoting=csv.QUOTE_ALL
#         df = pd.concat([df,tmp])
# #     index = index+1
# #     if (index==22) :
# #         break
```

### DATA


```python
# pickle the df to be used later 
#df.to_pickle(gen_data_PATH+'FEB_MARCH_dataframe.pkl')
#df_feb_march_ref = pd.read_pickle(gen_data_PATH+'feb_march_en_refugee_raw.pkl')
df_april= pd.read_pickle(gen_data_PATH+'Generated_Dataapril_dataframe.pkl.pkl')
```


```python
len(df_feb_march_ref)
```




    38310



## Catenate February, March and April months


```python
df_feb_march = pd.read_pickle(gen_data_PATH+'FEB_MARCH_dataframe.pkl')
#df_april = pd.read_pickle(gen_data_PATH+'Generated_Dataapril_dataframe.pkl')#
```


```python
#df = pd.concat([df_feb_march,df_april])
```

### Rows


```python
print(f'The FEBRUARY MARCH dataset contains {len(df_feb_march)} tweet entries')
```

    The FEBRUARY MARCH dataset contains 15583279 tweet entries



```python
df = df_feb_march
```


```python
print(f'The APRIL dataset contains {len(df_april)} tweet entries')
```

### Columns


```python
df.columns.to_list()
```




    ['Unnamed: 0',
     'userid',
     'username',
     'acctdesc',
     'location',
     'following',
     'followers',
     'totaltweets',
     'usercreatedts',
     'tweetid',
     'tweetcreatedts',
     'retweetcount',
     'text',
     'hashtags',
     'language',
     'coordinates',
     'favorite_count',
     'extractedts']



### Sample


```python

```


```python
df[['location','tweetid','tweetcreatedts','text','hashtags','language']].sample(5)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /var/folders/3v/cd3_rvz10bd1jd08x0c03tpr0000gt/T/ipykernel_4105/86441832.py in <module>
    ----> 1 df[['location','tweetid','tweetcreatedts','text','hashtags','language']].sample(5)
    

    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/pandas/core/frame.py in __getitem__(self, key)
       3515             indexer = np.where(indexer)[0]
       3516 
    -> 3517         data = self._take_with_is_copy(indexer, axis=1)
       3518 
       3519         if is_single_key:


    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/pandas/core/generic.py in _take_with_is_copy(self, indices, axis)
       3714         See the docstring of `take` for full explanation of the parameters.
       3715         """
    -> 3716         result = self.take(indices=indices, axis=axis)
       3717         # Maybe set copy if we didn't actually change the index.
       3718         if not result._get_axis(axis).equals(self._get_axis(axis)):


    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/pandas/core/generic.py in take(self, indices, axis, is_copy, **kwargs)
       3699         nv.validate_take((), kwargs)
       3700 
    -> 3701         self._consolidate_inplace()
       3702 
       3703         new_data = self._mgr.take(


    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/pandas/core/generic.py in _consolidate_inplace(self)
       5651             self._mgr = self._mgr.consolidate()
       5652 
    -> 5653         self._protect_consolidate(f)
       5654 
       5655     @final


    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/pandas/core/generic.py in _protect_consolidate(self, f)
       5639             return f()
       5640         blocks_before = len(self._mgr.blocks)
    -> 5641         result = f()
       5642         if len(self._mgr.blocks) != blocks_before:
       5643             self._clear_item_cache()


    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/pandas/core/generic.py in f()
       5649 
       5650         def f():
    -> 5651             self._mgr = self._mgr.consolidate()
       5652 
       5653         self._protect_consolidate(f)


    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/pandas/core/internals/managers.py in consolidate(self)
        629         bm = type(self)(self.blocks, self.axes, verify_integrity=False)
        630         bm._is_consolidated = False
    --> 631         bm._consolidate_inplace()
        632         return bm
        633 


    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/pandas/core/internals/managers.py in _consolidate_inplace(self)
       1683     def _consolidate_inplace(self) -> None:
       1684         if not self.is_consolidated():
    -> 1685             self.blocks = tuple(_consolidate(self.blocks))
       1686             self._is_consolidated = True
       1687             self._known_consolidated = True


    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/pandas/core/internals/managers.py in _consolidate(blocks)
       2082     new_blocks: list[Block] = []
       2083     for (_can_consolidate, dtype), group_blocks in grouper:
    -> 2084         merged_blocks = _merge_blocks(
       2085             list(group_blocks), dtype=dtype, can_consolidate=_can_consolidate
       2086         )


    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/pandas/core/internals/managers.py in _merge_blocks(blocks, dtype, can_consolidate)
       2109             # Sequence[Union[int, float, complex, str, bytes, generic]],
       2110             # Sequence[Sequence[Any]], SupportsArray]]
    -> 2111             new_values = np.vstack([b.values for b in blocks])  # type: ignore[misc]
       2112         else:
       2113             bvals = [blk.values for blk in blocks]


    <__array_function__ internals> in vstack(*args, **kwargs)


    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/numpy/core/shape_base.py in vstack(tup)
        281     if not isinstance(arrs, list):
        282         arrs = [arrs]
    --> 283     return _nx.concatenate(arrs, 0)
        284 
        285 


    <__array_function__ internals> in concatenate(*args, **kwargs)


    KeyboardInterrupt: 


## 2. Dataset selection
We want to keep only english and refugee related tweets


```python
# available languages 
lan = df['language'].value_counts().sort_values(ascending=False)[0:10]
```


```python
lan_per = lan.apply(lambda x : x/len(df)*100)
```


```python
# plot
plt = lan_per.plot.bar(rot=45,figsize=(15, 10),color=['C0', 'C1', 'C2','C3','C5','C6','C7','C8','C9','C10'])

# parameters
plt.set_ylabel('Percentage of tweets',fontdict={'fontsize':20})
plt.set_xlabel('Languages',fontdict={'fontsize':20})
plt.tick_params(axis='x', which='both', labelsize=22)
plt.tick_params(axis='y', which='both', labelsize=22)
plt.set_title('Language distribution of the April dataset',pad=20, fontdict={'fontsize':24})
```




    Text(0.5, 1.0, 'Language distribution of the April dataset')




    
![png](Descriptive-statistics_files/Descriptive-statistics_27_1.png)
    



```python
# english selection
df_en_raw = df[df['language']== 'en']
len_en_raw = len(df_en_raw['tweetid'])
# french selection
df_fr_raw = df[df['language']== 'fr']
len_fr_raw = len(df_fr_raw['tweetid'])
```

#### De-duplication


```python
df_en_raw.head()
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
      <th>Unnamed: 0</th>
      <th>userid</th>
      <th>username</th>
      <th>acctdesc</th>
      <th>location</th>
      <th>following</th>
      <th>followers</th>
      <th>totaltweets</th>
      <th>usercreatedts</th>
      <th>tweetid</th>
      <th>...</th>
      <th>original_tweet_id</th>
      <th>original_tweet_userid</th>
      <th>original_tweet_username</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>in_reply_to_screen_name</th>
      <th>is_quote_status</th>
      <th>quoted_status_id</th>
      <th>quoted_status_userid</th>
      <th>quoted_status_username</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>16882774</td>
      <td>Yaniela</td>
      <td>Animal lover, supports those who fight injustice wherever it raises its evil head. Personality flaws: Grumpy on occasion, cannot tolerate stupidity. #VOTEBLUE</td>
      <td>Hawaii</td>
      <td>1158</td>
      <td>392</td>
      <td>88366</td>
      <td>2008-10-21 07:34:04.000000</td>
      <td>1509681950042198030</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3205296069</td>
      <td>gregffff</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>122</td>
      <td>881</td>
      <td>99853</td>
      <td>2015-04-25 11:24:34.000000</td>
      <td>1509681950151348229</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1235940869812809728</td>
      <td>ThanapornThon17</td>
      <td>เล่นไวโอลิน\nพูดภาษาจีน</td>
      <td>NaN</td>
      <td>231</td>
      <td>72</td>
      <td>5481</td>
      <td>2020-03-06 14:52:01.000000</td>
      <td>1509681950683926556</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1347985375566966784</td>
      <td>I_Protest_2021</td>
      <td>01000001 01101110 01101111 01101110 01111001 01101101 01101111 01110101 01110011 00100001</td>
      <td>International Web Zone</td>
      <td>399</td>
      <td>377</td>
      <td>301</td>
      <td>2021-01-09 19:15:44.000000</td>
      <td>1509681951116046336</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1505394816636846083</td>
      <td>Marsh_Win_01</td>
      <td>🌿@Pickaw @TWITTERPICKER 🌿Winning isn’t everything🌾but wanting to win is🎊 #Marsh_Win_01 🌱</td>
      <td>Hunter Account</td>
      <td>158</td>
      <td>25</td>
      <td>8982</td>
      <td>2022-03-20 04:04:40.000000</td>
      <td>1509681951304990720</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>




```python
# To keep the original tweet, sort the dataframe by creation dates
df_en_raw = df_en_raw.sort_values(['tweetcreatedts'])
#df_en_raw.head(50)
#df_fr_raw = df_fr_raw.sort_values(['tweetcreatedts'])
```


```python
# english 
df_en_unique = df_en_raw[df_en_raw.duplicated(subset=['text'], keep='first')==False]# the duplciated work as a mask, the false are not duplicated
len_df_en_uni = len(df_en_unique[df_en_unique==False])
per_en_unique = (len_df_en_uni/len_en_raw)*100
print('{:.4}% of the english tweets are retweet. There are {} english uniques tweets'.format(per_en_unique,len_df_en_uni))
```

    24.73% of the english tweets are retweet. There are 2594755 english uniques tweets



```python
# french 
df_fr_unique = df_fr_raw[df_fr_raw.duplicated(subset=['text'], keep='first')==False] # the duplciated work as a mask, the false are not duplicated
len_df_fr_uni = len(df_fr_unique)
per_fr_unique = (len_df_fr_uni/len_fr_raw)*100
print('{:.4}% of the french tweets are retweet. There are {} french uniques tweets'.format(per_fr_unique,len_df_fr_uni))
```

### Dataset features


```python
df_en = df_en_unique
#df_fr = df_fr_unique
```


```python
per_en = len(df_en)/len(df)*100
print(f'There are {len(df_en)} unique english tweets.')
```

    There are 2594755 unique english tweets.



```python
per_fr = len(df_fr)/len(df)*100
print(f'There are {len(df_fr)} unique french tweets.')
```

    There are 76517 unique french tweets.



```python
keywords_en = ('refugee|migrant|asylum seeker')
keywords_fr = ('refugié|migrant|demandeur d\'asile')
```


```python
# English tweets with REFUGEE in TEXT
ref_en = df_en[df_en['text'].str.contains(keywords_en)]
per_en_ref = len(ref_en)/len(df_en)*100
print(f'There are {len(ref_en)} english tweets containing refugee keywords.')
print('{:2.3}% of the english tweets are related to the refugee topic.'.format(per_en_ref))
```

    There are 38310 english tweets containing refugee keywords.
    1.48% of the french tweets are related to the refugee topic.



```python
ref_en = pd.read_pickle(gen_data_PATH+'ref_en_unique_ref.pkl')
```


```python
# French tweets with REFUGEE in TEXT
ref_fr = df_fr[df_fr['text'].str.contains(keywords_fr)]
per_fr_ref = len(ref_fr)/len(df_fr)*100
print(f'There are {len(ref_fr)} french tweets containing refugee keywords.')
print('{:2.3}% of the french tweets are related to the refugee topic.'.format(per_fr_ref))
```

    There are 80 french tweets containing refugee keywords.
    0.105% of the french tweets are related to the refugee topic.


#### Sample


```python
ref_en[['location','tweetid','tweetcreatedts','text','hashtags','language']].sample(5)
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
      <th>location</th>
      <th>tweetid</th>
      <th>tweetcreatedts</th>
      <th>text</th>
      <th>hashtags</th>
      <th>language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>214650</th>
      <td>Web-based</td>
      <td>1513540001686040576</td>
      <td>2022-04-11 15:30:31.000000</td>
      <td>The entire A2L library and supporting resources will be made available at no cost to schools supporting refugees.\n\nClick to learn more: https://t.co/IW4ZrDieiA\n\n#ukraine  #refugeesupport #educationmatters #learningneverstops #helpspreadthenews htt...</td>
      <td>[{'text': 'ukraine', 'indices': [161, 169]}, {'text': 'refugeesupport', 'indices': [171, 186]}, {'text': 'educationmatters', 'indices': [187, 204]}, {'text': 'learningneverstops', 'indices': [205, 224]}, {'text': 'helpspreadthenews', 'indices': [225, ...</td>
      <td>en</td>
    </tr>
    <tr>
      <th>290340</th>
      <td>The #UK 🇬🇧 and #EU 🇪🇺</td>
      <td>1513938313429893127</td>
      <td>2022-04-12 17:53:16.000000</td>
      <td>@10DowningStreet @NadineDorries How many #Ukraine refugees have been given help in the Great #ENGLAND by this @Conservatives government?\n\nThe great British people have stepped up but absolutely no help by this government 🙄</td>
      <td>[{'text': 'Ukraine', 'indices': [41, 49]}, {'text': 'ENGLAND', 'indices': [93, 101]}]</td>
      <td>en</td>
    </tr>
    <tr>
      <th>100749</th>
      <td>NaN</td>
      <td>1510175909671837696</td>
      <td>2022-04-02 08:42:49.000000</td>
      <td>Putin's generals and commanders are also responsible for Russia's atrocities. Many of them have experience bombing Syria, so they are masters in turning people into refugees and beautiful cities into graveyards. #StopRussia https://t.co/rDcYhnaITk</td>
      <td>[]</td>
      <td>en</td>
    </tr>
    <tr>
      <th>104546</th>
      <td>Tamilnadu</td>
      <td>1511973401564696576</td>
      <td>2022-04-07 07:45:24.000000</td>
      <td>Spanish Kindergarten children welcoming a new refugee student from Ukraine\n\n#Ukraine #UkraineRussia #UkraineRussiaConflict #Spain #Kindergarden #healing #healingjourney #classmates #Newstn https://t.co/QD6KjpDXW7</td>
      <td>[{'text': 'Ukraine', 'indices': [76, 84]}, {'text': 'UkraineRussia', 'indices': [85, 99]}, {'text': 'UkraineRussiaConflict', 'indices': [100, 122]}, {'text': 'Spain', 'indices': [123, 129]}, {'text': 'Kindergarden', 'indices': [130, 143]}, {'text': 'h...</td>
      <td>en</td>
    </tr>
    <tr>
      <th>338722</th>
      <td>NaN</td>
      <td>1511786255033450512</td>
      <td>2022-04-06 19:21:45.000000</td>
      <td>A pleasure to talk with the Harvard EdCast &amp;amp; reflect on what we can learn from other experiences of #refugeeeducation for millions of children fleeing #Ukraine, drawing on research in #RightWhereWeBelong https://t.co/MKKXSagvfe</td>
      <td>[{'text': 'refugeeeducation', 'indices': [104, 121]}, {'text': 'Ukraine', 'indices': [155, 163]}, {'text': 'RightWhereWeBelong', 'indices': [188, 207]}]</td>
      <td>en</td>
    </tr>
  </tbody>
</table>
</div>




```python
ref_en[['text']].iloc[15]
```




    text    Anyone saying that the #Ukraine conflict started in Feb is either lying or is clueless. The world has millions of more refugees today because of the regime change &amp; #NATO expansion headed by the US in 2014. Not to mention the US using our tax doll...
    Name: 4286, dtype: object




```python
ref_fr[['location','tweetid','tweetcreatedts','text','hashtags','language']].sample(5)
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
      <th>location</th>
      <th>tweetid</th>
      <th>tweetcreatedts</th>
      <th>text</th>
      <th>hashtags</th>
      <th>language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65800</th>
      <td>NaN</td>
      <td>1515940589283463168</td>
      <td>2022-04-18 06:29:35.000000</td>
      <td>Merci #OTAN et #Europe\nD offrir des esclaves sexuelles blanches ukrainiennes pour satisfaire les colons noirs et arabes\n@MarleneSchiappa va encore  dire qu il faut aider #migrants et que tous ça c la faute a #Poutine\n#Ukraine #Islamisme #grandrempl...</td>
      <td>[{'text': 'OTAN', 'indices': [6, 11]}, {'text': 'Europe', 'indices': [15, 22]}, {'text': 'migrants', 'indices': [170, 179]}, {'text': 'Poutine', 'indices': [208, 216]}, {'text': 'Ukraine', 'indices': [217, 225]}, {'text': 'Islamisme', 'indices': [226,...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>70914</th>
      <td>Paris, France</td>
      <td>1515226390685265921</td>
      <td>2022-04-16 07:11:37.000000</td>
      <td>7...et dans la sénatoriale d'Ohio, JD Vance se fait le champion d'1 défense des travailleurs blancs qui seraient "remplacés" par des immigrants clandestins affluant à la frontière mexicaine délaissée pendant que #Biden se préoccupe à tort de l'#Ukrain...</td>
      <td>[{'text': 'Biden', 'indices': [212, 218]}, {'text': 'Ukraine', 'indices': [244, 252]}]</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>189803</th>
      <td>Toulouse (France)</td>
      <td>1514966571000737796</td>
      <td>2022-04-15 13:59:11.000000</td>
      <td>RT @CarodeCamaret: #UE : à l'#Est, tout est pardonné ? Débordés par les fluex de #migrants d'#Ukraine, la #Pologne et le #Hongrie peuvent-ils en oublier l'Etat de droit #libertedelapresse #justice @msojdrova @EPPGroup #Tchèque et @marctarabella @PES_P...</td>
      <td>[{'text': 'UE', 'indices': [19, 22]}, {'text': 'Est', 'indices': [29, 33]}, {'text': 'migrants', 'indices': [81, 90]}, {'text': 'Ukraine', 'indices': [93, 101]}, {'text': 'Pologne', 'indices': [106, 114]}, {'text': 'Hongrie', 'indices': [121, 129]}, {...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>135038</th>
      <td>Paris, France</td>
      <td>1511257641795923975</td>
      <td>2022-04-05 08:21:14.000000</td>
      <td>Guerre en #Ukraine : des demandeurs d’asile prisonniers au milieu des combats \n\nL’ONG Human Rights Watch @hrw alerte sur le sort de #migrants ou #réfugiés, toujours emprisonnés sous le feu des bombes. ⤵\nhttps://t.co/x6VmvNXANh</td>
      <td>[{'text': 'Ukraine', 'indices': [10, 18]}, {'text': 'migrants', 'indices': [132, 141]}, {'text': 'réfugiés', 'indices': [145, 154]}]</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>268326</th>
      <td>Hérault, Languedoc-Roussillon</td>
      <td>1513209348096507906</td>
      <td>2022-04-10 17:36:37.000000</td>
      <td>💍 42 ans de mariage pour Olga et Oleh, refugiés 🇺🇦 pris en charge par @CroixRouge sur le site d’hébergement mis en place par @montpellier_ @Prefet34 @CroixRouge34. Surprise! Nos équipes leur ont offert un joli 💐, des 🍫 et 1 énooorme 🎂, dégusté avec éq...</td>
      <td>[{'text': 'Ukraine', 'indices': [265, 273]}]</td>
      <td>fr</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Pre-processing for NLP

### Helpers 



```python
import re 
def clean_text(text):
    
    # remove line breaks
    text = text.replace('\n',' ')
    
    #remove URLS
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    text = re.sub(urlPattern,'URL',text)
    
    # Replace 3 or more consecutive letters by 2 letter.
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    text = re.sub(sequencePattern, seqReplacePattern, text)
    
     # Remove ampersand amp name
    text = re.sub('&amp;', 'and', text)
    
    # Remove unwanted symbols but preserve sentence structure by maintaining "?" , ",", "!", and "." and "#"
    # Note that this remove cyrilic characters
    alphaPattern      = "[^a-zA-Zàâçéèêëîïôûùüÿñæœ0-9?,!.#]"
    text = re.sub(alphaPattern, ' ', text)
    
    # Remove lonely # 
    text = re.sub(' # ', ' ', text)


    # remove long multiple whitespaces
    text = re.sub(r"( )\1\1+", ' ', text)
    
    return text
    
```


```python
len(ref_en)
```




    38310




```python
# cleaning
# english 
ref_en['text'] = ref_en['text'].swifter.apply(lambda x : clean_text(x))

#french
#ref_fr['text'] = ref_fr['text'].apply(lambda x : clean_text(x))
```

    /opt/anaconda3/envs/ada/lib/python3.8/site-packages/swifter/swifter.py:33: UserWarning: This pandas object has duplicate indices, and swifter may not be able to improve performance. Consider resetting the indices with `df.reset_index(drop=True)`.
      warnings.warn(



    Pandas Apply:   0%|          | 0/38310 [00:00<?, ?it/s]


    /var/folders/3v/cd3_rvz10bd1jd08x0c03tpr0000gt/T/ipykernel_4105/1549380104.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      ref_en['text'] = ref_en['text'].swifter.apply(lambda x : clean_text(x))



```python
ref_en.to_pickle(gen_data_PATH+'feb_march_refugee_clean.pkl')
#ref_en = pd.read_pickle(gen_data_PATH+'ref_en_unique_ref.pkl')
```


```python
# french 
ref_fr.to_pickle(gen_data_PATH+'ref_fr_unique_ref.pkl')
ref_fr = pd.read_pickle(gen_data_PATH+'ref_fr_unique_ref.pkl')
```


```python
ref_en[['location','text']].head()
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
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>328</th>
      <td>NaN</td>
      <td>Here s the link to our therapeutic tale for #refugee children from #Ukraine. In two language versions  UKR and RUS . PL and ENG translations available. #HelpUkraine  URL  Pleast, please RT</td>
    </tr>
    <tr>
      <th>364</th>
      <td>NaN</td>
      <td>Russian Nobel peace prize winner sells medal to fund for Ukrainian refugees #StopPutin #StandWithUkraine URL</td>
    </tr>
    <tr>
      <th>482</th>
      <td>NaN</td>
      <td>I see #ColmOGorman hasn t volunteered to look after, say, two disabled refugees from #Ukraine. Why not? Surly Colm knows how important it is for Secular Saints to lead from the front. He hasn t refused, has he? URL</td>
    </tr>
    <tr>
      <th>706</th>
      <td>Floriduh, USA</td>
      <td>#IMMIGRATION #Biden Admin lifts #Trump era #Title42 which W.H. admits will create  influx  of #migrants  URL</td>
    </tr>
    <tr>
      <th>859</th>
      <td>NaN</td>
      <td>A refugee from #Mariupol tells how the AFU #Azov fired at their car with children while they were trying to leave the city. #Ukraine URL</td>
    </tr>
  </tbody>
</table>
</div>




```python
ref_fr[['location','text']].head()
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
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31881</th>
      <td>NaN</td>
      <td>#Exode JC.#Fartoukh #VendrediLecture #FridayReads #Ukraine Brisés,agonisants,au milieu des gravats Ils ont maudit les dieux qui les ont Nous somme tous,potentiellement,des #migrants Nous sommes tous des tresHumains URL #lesecrits URL URL</td>
    </tr>
    <tr>
      <th>106805</th>
      <td>NaN</td>
      <td>PUBLICATION 2e note du comité d experts en sciences sociales CFDT Fondation  j jaures consacrée aux #migrants, par  marie saglio, Pauline Doyen,  DRouilleault Franceterdasile  et  CfdtBerger  URL #Ukraine #crisesanitaire #réfugiés URL</td>
    </tr>
    <tr>
      <th>121217</th>
      <td>Genève, Suisse</td>
      <td>En quatre semaines, plus de 3 millions de personnes ont quitté l Ukraine en direction de l espace Schengen. Catherine Guignard Aeby répond à neuf questions sur le statut S.  KPMG CH  #finance #ukraine #refugiés #legal #politique  URL</td>
    </tr>
    <tr>
      <th>125602</th>
      <td>Résiste.Prouve que tu existes.</td>
      <td>Je suis totalement opposée â la livraison d armes à l #Ukraine et totalement opposée à la livraison de migrants ukrainiens à la France.</td>
    </tr>
    <tr>
      <th>133493</th>
      <td>Bordeaux</td>
      <td>C est le problème avec les confinements successifs, les gens ont trop pris le plis d être servis à domicile.  #Ukraine #réfugiés #migrants #RacismeDécomplexé #DessinDePresse #cartoonist URL</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(ref_en)
```




    12630



#### Do we want to de-emojify?
#### Do we really want to keep the hashtags?

#### Add swifter

### Date distribution
Tweet created per day /!\ not the same as extracted per date


```python
# to datetime format
ref_en['tweetcreatedts'] = pd.to_datetime(ref_en['tweetcreatedts'])
ref_fr['tweetcreatedts'] = pd.to_datetime(ref_fr['tweetcreatedts'])
```


```python
# data to aggregate 
ref_eng_agg = ref_en[['tweetcreatedts','userid']]

# group data per day and count
ref_en_daily_agg= ref_eng_agg.groupby(pd.Grouper(key='tweetcreatedts', axis=0, freq='D')).count()
```


```python
# plot per day 
ax = ref_en_daily_agg.plot(kind='bar', grid=False, figsize=(16,8), color='#586994', zorder=2, rot=15)
ax.set_xlabel('Date of creation',fontsize=22)
ax.set_ylabel('Numbers of tweets',fontsize=22)
ax.set_title('Daily english Tweets about the refugee crisis',fontsize=22)
```




    Text(0.5, 1.0, 'Daily english Tweets about the refugee crisis')




    
![png](Descriptive-statistics_files/Descriptive-statistics_61_1.png)
    



```python
# french 

# data to aggregate 
ref_fr_agg = ref_fr[['tweetcreatedts','userid']]

# group data per day and count
ref_fr_daily_agg= ref_fr_agg.groupby(pd.Grouper(key='tweetcreatedts', axis=0, freq='D')).count()
```


```python
# plot per day 
ax = ref_fr_daily_agg.plot(kind='bar', grid=False, figsize=(16,8), color='#586994', zorder=2, rot=15)
ax.set_xlabel('Date of creation',fontsize=22)
ax.set_ylabel('Numbers of tweets',fontsize=22)
ax.set_title('Daily french Tweets about the refugee crisis',fontsize=22)
```




    Text(0.5, 1.0, 'Daily french Tweets about the refugee crisis')




    
![png](Descriptive-statistics_files/Descriptive-statistics_63_1.png)
    


### Location distribution


```python
# string type 
ref_fr['location']= ref_fr['location'].astype('string')
ref_fr['location'].fillna('unknown',inplace=True)

ref_en['location']= ref_en['location'].astype('string')
ref_en['location'].fillna('unknown',inplace=True)
```


```python
# Locations are manually entered, we need to clean them for comparison ('London', 'london', 'london,england' should be grouped together)
def clean_location(loc):
    loc = loc.lower()
    loc_list = loc.split(',')
    loc = loc_list[0]
    return loc
```


```python
ref_en['location'].iloc[1].lower()
```




    'unknown'




```python
ref_en['location'] = ref_en['location'].apply(lambda x: clean_location(x))
```


```python
# plot 
ref_en_loc = ref_en.location.value_counts()[:20].plot.bar(figsize = (16,8),rot=45,color='#009384')

# parameters
ref_en_loc.set_xlabel('Location')
ref_en_loc.set_ylabel('Number of tweets')
ref_en_loc.set_title('Location of english tweets about the refugee crisis')

```




    Text(0.5, 1.0, 'Location of english tweets about the refugee crisis')




    
![png](Descriptive-statistics_files/Descriptive-statistics_69_1.png)
    



```python
# we keep the 20 first locations and group them by regions if they are part of the same region 
def merge_per_region(location):
    new_location = location 
    
    if (location == 'ukraine') or (location == 'kiev'):
        new_location = 'ukrania'
    
    if (location == 'uk') or (location == 'london') or (location =='england'):
        new_location = 'united kingdom'
    
    if (location == 'new york') or (location == 'washington') or (location == 'los angeles') or (location == 'usa'):
        new_location = 'ukrania':
        new_location = 'united states'
        
    if (location == 'toronto') :
        new_location = 'canada'
    
     if (location == 'global') :
        new_location = 'worldwide'
    
    return new_location
```


```python
# subset without unknwon 

ref_en_with_loc = ref_en[ref_en['location']!='unknown']
# plot 
plot = ref_en_with_loc.location.value_counts()[:20].plot.bar(figsize = (16,8),rot=45,color='#009384')

# parameters
plot.set_xlabel('Location')
plot.set_ylabel('Number of tweets')
plot.set_title('Location of english tweets about the refugee crisis')

```




    Text(0.5, 1.0, 'Location of english tweets about the refugee crisis')




    
![png](Descriptive-statistics_files/Descriptive-statistics_71_1.png)
    



```python
unkwn_loc_en = ref_en[ref_en['location']=='unknown']
per_unknown_en = len(unkwn_loc_en)/len(ref_en)*100
print('{:.4}% of location in the english tweets are unknown.'.format(per_unknown_en))
```

    27.13% of location in the english tweets are unknown.



```python
# french 
#ref_fr['location'] = ref_fr[ref_fr['location']== None].apply(lambda x :  x = '')
#ref_fr['location'] = ref_fr['location'].apply(lambda x : x.lower)

# plot 
ref_fr_loc = ref_fr.location.value_counts()[:20].plot.bar(figsize = (16,8),rot=45,color='#009384')

# parameters
ref_en_loc.set_xlabel('Location')
ref_en_loc.set_ylabel('Number of tweets')
ref_en_loc.set_title('Location of french tweets about the refugee crisis')

```




    Text(0.5, 1.0, 'Location of french tweets about the refugee crisis')




    
![png](Descriptive-statistics_files/Descriptive-statistics_73_1.png)
    


### Free up resources


```python

```

### 4. Words frequency

Should we look at bigrams and trigrams?


```python
# concatenate all texts cells 
all_ref_en = ref_en['text'].values.sum()

#Export all the tweets of interest in a single text file
text_file = open(gen_data_PATH+"all_tweets_en.txt", "w")
text_file.write(all_ref_en)
text_file.close()
```


```python
#get rid of common english words
en_stopwords = nltk.corpus.stopwords.words('english') #list of words such as a, the, and etc..
     

# FUNCTION
def ngram_frequency(text,stopwords):
    '''
    Description: Counting the frequency of n-grams in the text
    Input: A single string containing the text of interest 
    Output: List of bigram and their counts in the text in the format ((string,string),integer)
    Requirement: Nltk with stopwords, Counter 
    Use: this function is set to find bigrams, it can be extended for other n-grams
    '''
    
    #separate the text into words 
    allWords = nltk.tokenize.word_tokenize(text) 
    
    #gets rid on 1-letter words and 2-letters words
    allLongWords = []
    for word in allWords:
        if len(word) > 2: 
            allLongWords.append(word)   
   
    allWordExceptStop =[]
    for w in allLongWords:
        if w.lower() not in stopwords:
            allWordExceptStop.append(w)
    #create a list of bigrams words in the text. Can be adapted to n-grams zipping more words
    onegrams = allWordExceptStop
    #calculate the frequency of each bigram 
    onegramsFreq = nltk.FreqDist(onegrams) 
    
    #bigrams = zip(allWordExceptStop, allWordExceptStop[1:])
    #calculate the frequency of each bigram 
    #bigramsFreq = nltk.FreqDist(bigrams) 
    return onegramsFreq
```


```python
# english 

#Couting onegram frequencies for all articles of interest
onegramFreq_en = ngram_frequency(all_ref_en)

MAX = 10

#Visualize the most common bigrams
for word, frequency in onegramFreq_en.most_common(MAX):
        print('%s;%d' % (word, frequency))
```

    refugees;10197
    Ukraine;9795
    URL;5976
    Ukrainian;3604
    refugee;2060
    war;1979
    help;1927
    people;1804
    Russia;1702
    support;1547



```python
most_com_eng = pd.DataFrame(onegramFreq_en.most_common(MAX)).sort_values(by=[1],ascending=True)
plot = most_com_eng.set_index([0]).plot.barh(rot=25,color = '#b85741')

# parameters
plot.set_xlabel('Number of occurences',fontsize=16)
plot.set_ylabel('')
plot.set_title('Onegram frequency in the english dataset',fontsize=16)
plot.tick_params(axis='x', which='both', labelsize=14)
plot.tick_params(axis='y', which='both', labelsize=14)
```


    
![png](Descriptive-statistics_files/Descriptive-statistics_81_0.png)
    


stopwords(, str lower, remove urls,  no emoji, unescape? see https://www.kaggle.com/code/bwandowando/generate-wordcloud-from-english-tweets)

### ? Do we want to do TOPIC MODELING too?

### SCRATCH
some twitter tips from the assignemetn

## OLD FRENCH CODE


```python
# concatenate all texts cells 
all_ref_fr = ref_fr['text'].values.sum()

#Export all the tweets of interest in a single text file
text_file = open(gen_data_PATH+"all_tweets_fr.txt", "w")
text_file.write(all_ref_fr)
text_file.close()
```


```python
# french 
#Couting bigram frequencies for all articles of interest
onegramFreq_fr = ngram_frequency(all_ref_fr)

MAX = 10

#Visualize the most common bigrams
for word, frequency in onegramFreq_fr.most_common(MAX):
        print('%s;%d' % (word, frequency))
```

    les;67
    migrants;61
    Ukraine;58
    URL;49
    des;46
    pour;27
    réfugiés;23
    que;21
    sur;20
    est;18



```python
most_com_fr = pd.DataFrame(onegramFreq_fr.most_common(MAX))
most_com_fr.set_index([0]).plot.bar(rot=25)
```




    <AxesSubplot:xlabel='0'>




    
![png](Descriptive-statistics_files/Descriptive-statistics_88_1.png)
    



```python
import matplotlib.pyplot as plt
# Distribution of language declared in tweet metadata
fig,ax = plt.subplots(figsize=(18, 10))
plt = data['lang'].hist(figsize=(18,10),bins = 23)
ax.set_xlabel('Languages',fontsize =16)
ax.set_ylabel('Number of tweets',fontsize =16)

```




    Text(0, 0.5, 'Number of tweets')




    
![png](Descriptive-statistics_files/Descriptive-statistics_89_1.png)
    



```python

```


```python
## English
eng= len(df[df['lang']=='en'])/(len(df))*100
print(f'English is {eng} percent')
## German
de = len(df[df['lang']=='de'])/(len(df))*100
print(f'German is {de} percent')

## French
de = len(df[df['lang']=='fr'])/(len(df))*100
print(f'French is {de} percent')

## Italian
de = len(df[df['lang']=='it'])/(len(df))*100
print(f'Italian is {de} percent')

## Mandarin
de = len(df[df['lang']=='cmn'])/(len(df))*100
print(f'Mandarin chinese is {de} percent')

## Mandarin
de = len(df[df['lang']=='zh'])/(len(df))*100
print(f'Mandarin chinese is {de} percent')


## Hindi
hi = len(df[df['lang']=='hi'])/(len(df))*100
print(f'Hindi chinese is {hi} percent')

## Spanish
es = len(df[df['lang']=='es'])/(len(df))*100
print(f'Espagnol  is {es} percent')


## Arabic
ar = len(df[df['lang']=='ar'])/(len(df))*100
print(f'Arabic  is {ar} percent')
```

    English is 82.89999999999999 percent
    German is 1.1666666666666667 percent
    French is 6.953333333333333 percent
    Italian is 0.18 percent
    Mandarin chinese is 0.0 percent
    Mandarin chinese is 0.0 percent
    Hindi chinese is 0.12 percent
    Espagnol  is 0.26666666666666666 percent
    Arabic  is 0.0 percent



```python
n = len(data['lang'].unique())
print(f'There are {n} differents languages in our tweets\n with a majority of english tweets.')
```

    There are 27 differents languages in our tweets
     with a majority of english tweets.



```python
# Frequency of hashtags
hashdata = pd.concat([data.drop(['hashtags'], axis=1), data['hashtags'].apply(pd.Series)], axis=1)
hashdata.head()
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
      <th>retweeted_status</th>
      <th>lang</th>
      <th>urls</th>
      <th>user_mentions</th>
      <th>symbols</th>
      <th>media</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'created_at': 'Sat Apr 09 15:53:40 +0000 2022', 'id': 1512821053201997824, 'id_str': '1512821053201997824', 'text': 'First #beachclean of the year ✅ Around 20 volunteers came &amp;amp;we filled 3 big bags of #PlasticPollution, old fishing g… https://t.co...</td>
      <td>en</td>
      <td>[]</td>
      <td>[{'screen_name': 'LeosAnimalPlan1', 'name': 'Leo's Animal Planet #STOPTHEGRIND 🐬🐳', 'id': 1369998441531727879, 'id_str': '1369998441531727879', 'indices': [3, 19]}]</td>
      <td>[]</td>
      <td>NaN</td>
      <td>{'text': 'beachclean', 'indices': [27, 38]}</td>
      <td>{'text': 'PlasticPollution', 'indices': [108, 125]}</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'created_at': 'Sat Apr 09 11:25:01 +0000 2022', 'id': 1512753444037304321, 'id_str': '1512753444037304321', 'text': 'With our beautiful pink banner in Hyde Park! #EndFossilFuelsNow #JustStopOil #ClimateCrisis 
#ActNow… https://t.co/uNV7qN8LGx', 'disp...</td>
      <td>en</td>
      <td>[]</td>
      <td>[{'screen_name': 'XRWorthing', 'name': 'XR Worthing', 'id': 1146461945882664961, 'id_str': '1146461945882664961', 'indices': [3, 14]}]</td>
      <td>[]</td>
      <td>NaN</td>
      <td>{'text': 'EndFossilFuelsNow', 'indices': [61, 79]}</td>
      <td>{'text': 'JustStopOil', 'indices': [80, 92]}</td>
      <td>{'text': 'ClimateCrisis', 'indices': [93, 107]}</td>
      <td>{'text': 'ActNow', 'indices': [109, 116]}</td>
      <td>{'text': 'ExtinctionRebellion', 'indices': [117, 137]}</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>en</td>
      <td>[{'url': 'https://t.co/WjIg5MUUWz', 'expanded_url': 'https://twitter.com/i/web/status/1512842167798034438', 'display_url': 'twitter.com/i/web/status/1…', 'indices': [116, 139]}]</td>
      <td>[{'screen_name': 'lisamurkowski', 'name': 'Sen. Lisa Murkowski', 'id': 18061669, 'id_str': '18061669', 'indices': [0, 14]}, {'screen_name': 'Sen_JoeManchin', 'name': 'Senator Joe Manchin', 'id': 234374703, 'id_str': '234374703', 'indices': [15, 30]}, ...</td>
      <td>[]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>en</td>
      <td>[{'url': 'https://t.co/8IhHLu4iOD', 'expanded_url': 'https://twitter.com/i/web/status/1512842175842861060', 'display_url': 'twitter.com/i/web/status/1…', 'indices': [116, 139]}]</td>
      <td>[{'screen_name': 'edhollett', 'name': 'Edward Hollett', 'id': 17963341, 'id_str': '17963341', 'indices': [0, 10]}]</td>
      <td>[]</td>
      <td>NaN</td>
      <td>{'text': 'BayDuNord', 'indices': [70, 80]}</td>
      <td>{'text': 'ClimateChange', 'indices': [93, 107]}</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'created_at': 'Fri Mar 25 02:18:43 +0000 2022', 'id': 1507180146285072388, 'id_str': '1507180146285072388', 'text': '@IndivisibleMNLo @GovTimWalz .@IndivisibleMNLo and @WomensMarchMN both stopped their zoom recordings before the wor… https://t.co/E0z...</td>
      <td>en</td>
      <td>[]</td>
      <td>[{'screen_name': 'MNSnarkDept', 'name': 'MN Department Of Snark', 'id': 1418211881337069575, 'id_str': '1418211881337069575', 'indices': [3, 15]}, {'screen_name': 'IndivisibleMNLo', 'name': 'Indivisible MNLeg', 'id': 831517301312733184, 'id_str': '831...</td>
      <td>[]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
wordsList = []

def get_text(i):
    if (data.iloc[i]['hashtags']!=None):
        text = data.iloc[i]['hashtags'][0]['text'] 
        #print(text)
        wordsList.append(text)

```


```python
get_text(3)
```


```python
wordsList
```




    ['BayDuNord']




```python
# counts words for each row
for i in range(len(data)):
     get_text(i)
```


```python
dic_counts = dict(counts)
```


```python
values = pd.DataFrame.from_dict(dic_counts,orient='index')
values.head()
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BayDuNord</th>
      <td>3</td>
    </tr>
    <tr>
      <th>beachclean</th>
      <td>4</td>
    </tr>
    <tr>
      <th>EndFossilFuelsNow</th>
      <td>3</td>
    </tr>
    <tr>
      <th>ExtinctionRebellion</th>
      <td>32</td>
    </tr>
    <tr>
      <th>ClimateCrisis</th>
      <td>2194</td>
    </tr>
  </tbody>
</table>
</div>




```python
values = values.rename(columns={0:'Frequency'})
values = values.sort_values('Frequency',ascending=False)
values
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
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ClimateCrisis</th>
      <td>2194</td>
    </tr>
    <tr>
      <th>LetTheEarthBreath</th>
      <td>1158</td>
    </tr>
    <tr>
      <th>GIEC</th>
      <td>536</td>
    </tr>
    <tr>
      <th>climatechange</th>
      <td>511</td>
    </tr>
    <tr>
      <th>IPCC</th>
      <td>260</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>LNP</th>
      <td>1</td>
    </tr>
    <tr>
      <th>CoalBaronBlockade</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Netflix</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Asian</th>
      <td>1</td>
    </tr>
    <tr>
      <th>SaveTheDate</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>792 rows × 1 columns</p>
</div>




```python
values['Rank'] = np.arange(values.shape[0])
```


```python
values.index.name = 'Hashtags'
values
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
      <th>Frequency</th>
      <th>Rank</th>
    </tr>
    <tr>
      <th>Hashtags</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ClimateCrisis</th>
      <td>2194</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LetTheEarthBreath</th>
      <td>1158</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GIEC</th>
      <td>536</td>
      <td>2</td>
    </tr>
    <tr>
      <th>climatechange</th>
      <td>511</td>
      <td>3</td>
    </tr>
    <tr>
      <th>IPCC</th>
      <td>260</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>LNP</th>
      <td>1</td>
      <td>787</td>
    </tr>
    <tr>
      <th>CoalBaronBlockade</th>
      <td>1</td>
      <td>788</td>
    </tr>
    <tr>
      <th>Netflix</th>
      <td>1</td>
      <td>789</td>
    </tr>
    <tr>
      <th>Asian</th>
      <td>1</td>
      <td>790</td>
    </tr>
    <tr>
      <th>SaveTheDate</th>
      <td>1</td>
      <td>791</td>
    </tr>
  </tbody>
</table>
<p>792 rows × 2 columns</p>
</div>




```python
values.reset_index(inplace=True)
```


```python
values.drop(columns = ['Rank'],inplace=True)
```


```python
values.index.name = '# Rank'
values.head(30)
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
      <th>Hashtags</th>
      <th>Frequency</th>
    </tr>
    <tr>
      <th># Rank</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ClimateCrisis</td>
      <td>2194</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LetTheEarthBreath</td>
      <td>1158</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GIEC</td>
      <td>536</td>
    </tr>
    <tr>
      <th>3</th>
      <td>climatechange</td>
      <td>511</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IPCC</td>
      <td>260</td>
    </tr>
    <tr>
      <th>5</th>
      <td>COP26</td>
      <td>247</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ClimateChange</td>
      <td>192</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ClimateAction</td>
      <td>186</td>
    </tr>
    <tr>
      <th>8</th>
      <td>scientistprotest</td>
      <td>154</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LetTheEarthBreathe</td>
      <td>137</td>
    </tr>
    <tr>
      <th>10</th>
      <td>climat</td>
      <td>115</td>
    </tr>
    <tr>
      <th>11</th>
      <td>solar</td>
      <td>87</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ClimateEmergency</td>
      <td>75</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ClimateActionNow</td>
      <td>62</td>
    </tr>
    <tr>
      <th>14</th>
      <td>climate</td>
      <td>53</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ClimateReport</td>
      <td>51</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Scientistprotest</td>
      <td>43</td>
    </tr>
    <tr>
      <th>17</th>
      <td>climatecrisis</td>
      <td>37</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ActOnClimate</td>
      <td>36</td>
    </tr>
    <tr>
      <th>19</th>
      <td>spreadawareness</td>
      <td>35</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Climatechange</td>
      <td>34</td>
    </tr>
    <tr>
      <th>21</th>
      <td>SaveSoil</td>
      <td>33</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ExtinctionRebellion</td>
      <td>32</td>
    </tr>
    <tr>
      <th>23</th>
      <td>FridaysForFuture</td>
      <td>29</td>
    </tr>
    <tr>
      <th>24</th>
      <td>ClimateJustice</td>
      <td>26</td>
    </tr>
    <tr>
      <th>25</th>
      <td>climateaction</td>
      <td>25</td>
    </tr>
    <tr>
      <th>26</th>
      <td>savetheearth</td>
      <td>24</td>
    </tr>
    <tr>
      <th>27</th>
      <td>auspol</td>
      <td>20</td>
    </tr>
    <tr>
      <th>28</th>
      <td>FossilFuels</td>
      <td>19</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ElectricVehicles</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
values.head(30).to_csv(index=False)
```




    'Hashtags,Frequency\nClimateCrisis,2194\nLetTheEarthBreath,1158\nGIEC,536\nclimatechange,511\nIPCC,260\nCOP26,247\nClimateChange,192\nClimateAction,186\nscientistprotest,154\nLetTheEarthBreathe,137\nclimat,115\nsolar,87\nClimateEmergency,75\nClimateActionNow,62\nclimate,53\nClimateReport,51\nScientistprotest,43\nclimatecrisis,37\nActOnClimate,36\nspreadawareness,35\nClimatechange,34\nSaveSoil,33\nExtinctionRebellion,32\nFridaysForFuture,29\nClimateJustice,26\nclimateaction,25\nsavetheearth,24\nauspol,20\nFossilFuels,19\nElectricVehicles,19\n'



## 5. Handles


```python
# Percentage of tweets directly generated by all the 20 media accounts
#together.
media_list
```




    ['@nytimes',
     '@TelegraphNews',
     '@GuardianNews',
     '@Newsweek',
     '@BBCAfrica',
     '@Independent',
     '@FRANCE24',
     '@CNBC',
     '@politico',
     '@SkyNewsBreak',
     '@AJENews',
     '@FT',
     '@BreakingNew',
     '@SkyNews',
     '@NDTVFeed',
     '@guardian',
     '@HuffPost',
     '@XHNews',
     '@AP',
     '@ABC']




```python
#get user data into a data frame 
def get_user(i):
    if (df.iloc[i]['user']!=None):
        name = df.iloc[i]['user']['name'] 
        return name

```


```python
df['username'] = df['user'].apply(lambda x: x['name'])
```


```python
df[['id','text','username']]
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
      <th>id</th>
      <th>text</th>
      <th>username</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1512842143336910849</td>
      <td>RT @LeosAnimalPlan1: First #beachclean of the year ✅ Around 20 volunteers came &amp;amp;we filled 3 big bags of #PlasticPollution, old fishing gear…</td>
      <td>Pam Sharman 🌻🌻🌻🌻🇺🇦🇺🇦🇺🇦🌠🌠</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1512842152811937795</td>
      <td>RT @XRWorthing: With our beautiful pink banner in Hyde Park! #EndFossilFuelsNow #JustStopOil #ClimateCrisis \n#ActNow #ExtinctionRebellion h…</td>
      <td>Things have to change! - Rejoin &amp; Reform</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1512842167798034438</td>
      <td>@lisamurkowski @Sen_JoeManchin @AESymposium Here’s a big idea:\nDeclare your support for making the Arctic a Global… https://t.co/WjIg5MUUWz</td>
      <td>salty seadude 🇺🇸Defend Democracy &amp; Environment</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1512842175842861060</td>
      <td>@edhollett You ARE wrong. We need to stop using fossil fuels now.\nBDN #BayDuNord will add to #ClimateChange.\n\nMost… https://t.co/8IhHLu4iOD</td>
      <td>Mark Dolore</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1512842180154773505</td>
      <td>RT @MNSnarkDept: @IndivisibleMNLo @GovTimWalz .@IndivisibleMNLo and @WomensMarchMN both stopped their zoom recordings before the worst of w…</td>
      <td>Joni Skibo/LaCroix❤️🇺🇦💪</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>1514604074091495439</td>
      <td>RT @JimGumboc: Why did they arrest the scientists for speaking up? This quote from Don't Look UP is so apt:\n“How is it criminal if we just…</td>
      <td>liu qingge's</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1514604074297344000</td>
      <td>https://t.co/38fObpZyDU</td>
      <td>‼️ LET THE EARTH BREATH ‼️</td>
    </tr>
    <tr>
      <th>997</th>
      <td>1514604074599034881</td>
      <td>@_GlobalCrisis_ I'm afraid to imagine the magnitude of the consequences of a 10-ball #earthquake in #Japan, in a ci… https://t.co/by7fhBz8br</td>
      <td>Vieda</td>
    </tr>
    <tr>
      <th>998</th>
      <td>1514604075102339078</td>
      <td>RT @LHS4LIF3R: “What can we do to help as a normal person?” #LetTheEarthBreath #ScientistRebellion #scientistsprotest #ClimateCrisis /c htt…</td>
      <td>🍓L(ia) ≷ Cas</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1514604075551055875</td>
      <td>RT @ksnoue: here are some ways to help!!\n\n#LetTheEarthBreath \n#scientistprotest \n#ClimateCrisis https://t.co/MPBGYax8ty</td>
      <td>soleil 🫧 let the earth breathe</td>
    </tr>
  </tbody>
</table>
<p>15000 rows × 3 columns</p>
</div>




```python
media_list_names =[]
for string in media_list:
    media_list_names.append(string[1:])
media_list_names 
```




    ['nytimes',
     'TelegraphNews',
     'GuardianNews',
     'Newsweek',
     'BBCAfrica',
     'Independent',
     'FRANCE24',
     'CNBC',
     'politico',
     'SkyNewsBreak',
     'AJENews',
     'FT',
     'BreakingNew',
     'SkyNews',
     'NDTVFeed',
     'guardian',
     'HuffPost',
     'XHNews',
     'AP',
     'ABC']




```python
# select tweet if names are in list 

not_retweet = df[df['retweeted_status'].isna()]

in_medias = not_retweet[not_retweet['username'].isin(media_list_names)]
per_medias = len(in_medias)/len(not_retweet)
print(f'There are {per_medias}% of direct tweets from the media list')
```

    There are 0.0% of direct tweets from the media list



```python
#cleaning ngo list
ngo_list_names =[]
for string in ngo_list:
    ngo_list_names.append(string[1:])
ngo_list_names 

# select tweet if names are in list 
in_ngos = not_retweet[not_retweet['username'].isin(ngo_list_names)]
per_ngos = len(in_ngos)/len(not_retweet)
print(f'There are {per_ngos}% of direct tweets from the ngo list')
```

    There are 0.0% of direct tweets from the ngo list



```python
#Percentage of tweets generated by all the 20 media accounts that appear as retweets in the sample.
retweet = df[df['retweeted_status'].isna() == False]

in_medias = retweet[retweet['username'].isin(media_list_names)]
per_medias = len(in_medias)/len(retweet)
print(f'There are {per_medias}% of retweets from the media list')

```

    There are 0.0% of retweets from the media list



```python
# select tweet if names are in list 
in_ngos = retweet[retweet['username'].isin(ngo_list_names)]
per_ngos = len(in_ngos)/len(retweet)
print(f'There are {per_ngos}% of retweets from the ngo list')
```

    There are 0.0% of retweets from the ngo list


## 6. Discussion

Point 4:  
The resuts are coherent with my initial hypothesis,namely that it would be mostly an english corpus. I was slightly surprised by the importance of retweet as I am not a user of twitter but it makes sense with respect to what we have learn in class.  

Point 5: 
Well I am wondering it there is an error in my code, as none of the tweets are coming or are retweeted by the media and NGO I've selectionned. However some sample are really small, for example the non retweeted samples are 2724 tweets, as there are  400 million users it then makes sense that only 20 handles are not even represented in the dataset. It is slightly more suprising for the larger datset of 12'276 tweets.


```python
len(not_retweet)
```




    2724




```python
len(retweet)
```




    12276




```python

```
