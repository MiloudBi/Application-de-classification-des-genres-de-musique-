import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import librosa
import pandas as pd






tracks = np.array(pd.read_csv(r"C:\Users\HP\Desktop\donneF.csv", low_memory=False))
#print(tracks[:,4:])
def feature():
    return tracks[:,4:]
#print(feature())
mylist=[]
for a in tracks[:,1] :
    if a not in mylist:
        mylist.append(a)
#print(mylist)
encoder = LabelEncoder()
y = encoder.fit_transform(mylist)
#print(y)

def labeel():
    r=[]
    for a in range(len(tracks)) :
        t=np.zeros((1,8))
        t[0,y[mylist.index(tracks[a,1])]]=1
        r.append(t)
    return np.concatenate(r)
        
"""
def labeelS():
    t=np.zeros(len(tracks))
    for a in range(len(tracks)) :
        t[a]=y[mylist.index(tracks[a,1])]
    return t


print(labeel().shape)
unique, counts = np.unique(tracks[:,1], return_counts=True)
print(dict(zip(unique, counts)))


elements =[]
for file in glob.glob(path):
    for filename in glob.glob(file+'\*.mp3'):
        elements.append(process(filename.split('\\')[-1][:6]))
        #print(process(filename.split('\\')[-1][:6]))
print(len(elements))
myl=[]
for i in range(len(small)):
    if small[i] in elements :
        myl.append(i)
#print(tracks.iloc[myl,:])
"""

#tra=trad[trad['set.1']=='small'][['track.7']]

#result = pd.concat((tra,tracks),axis=1)
#result = pd.concat((trad[['track.7','set.1']].iloc[2:,:],tracks),axis=1)
"""
trad=np.array(trad[['track.7','set.1']])
trad=trad[2:,:]
tracks=np.array(tracks)
tracks=tracks[3:,:]
result=np.concatenate((tracks,trad ),axis=1)
print(result[result[:,-1]=='small'].shape)
"""
#pd.DataFrame(result[result[:,-1]=='small']).to_csv(r"C:\Users\HP\Desktop\file.csv")

"""
mylist=[]
for a in tra['track.7'] :
    if a not in mylist:
        mylist.append(a)
print(mylist)
  """              

#print(process(list('000002')))       
