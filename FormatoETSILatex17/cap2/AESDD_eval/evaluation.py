#%%
from glob import glob
import pandas as pd


df=pd.DataFrame(glob('./AESDD/*/*.wav'),columns=['paths'])
df.head()
#%%
# path is like AESDD/label/filename.wav

df['labels']=df.paths.apply(lambda x: x.split('/')[2])
df.head()
# %%
from transformers import pipeline

pipe = pipeline("audio-classification", model="antonjaragon/emotions_6_classes_small")

# %%
# check if gpu is available
import torch
torch.cuda.is_available()

# %%
# get predictions
# predictions = [pipe(path)[0]['label'] for path in df.paths]
predictions = []
for path in df.paths:
    # print(path)
    # print(pipe(path)[0]['label'])
    try:
        predictions.append(pipe(path)[0]['label'])
    except:
        print(f'Error with {path}')
        predictions.append('Neutral')

# %%

TRANSLATIONS = {
    'Angry': 'anger',
    'Disgusted': 'disgust',
    'Fearful': 'fear',
    'Happy': 'happiness',
    'Sad': 'sadness',
    'Neutral': 'neutral'
}

y_pred = [TRANSLATIONS[pred] for pred in predictions]
y_true = df.labels.values


# %%
# save to pickle
import pandas as pd

df = pd.DataFrame({'y_pred':y_pred,'y_true':y_true})

# delete rows where y_pred is neutral
print(len(df))
df = df[df.y_pred != 'neutral']
print(len(df))

import pickle
with open('predictions.pkl','wb') as f:
    pickle.dump(df,f)


# %%
# load pickle
import pickle

with open('predictions.pkl','rb') as f:
    df_read=pickle.load(f)

# %%
