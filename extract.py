

import glob
import pandas as pd
import numpy as np

import os
import multiprocessing
import warnings

from tqdm import tqdm
import numpy as np
from scipy import stats
import pandas as pd
import librosa

import utils









def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()



def compute_feature(filepath):
        
        features = pd.Series(index=columns(), dtype=np.float32, name="test")
        def feature_stats(name, values):
            features[name, 'mean'] = np.mean(values, axis=1)
            features[name, 'std'] = np.std(values, axis=1)
            features[name, 'skew'] = stats.skew(values, axis=1)
            features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
            features[name, 'median'] = np.median(values, axis=1)
            features[name, 'min'] = np.min(values, axis=1)
            features[name, 'max'] = np.max(values, axis=1)
        try:
            x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast

            f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
            feature_stats('zcr', f)
            
            cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                     n_bins=7*12, tuning=None))
            assert cqt.shape[0] == 7 * 12
            assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1
            
            f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
            feature_stats('chroma_cqt', f)
            f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
            feature_stats('chroma_cens', f)
            f = librosa.feature.tonnetz(chroma=f)
            feature_stats('tonnetz', f)
            
            del cqt
            stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
            assert stft.shape[0] == 1 + 2048 // 2
            assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
            del x
            
            f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
            feature_stats('chroma_stft', f)
            
            f = librosa.feature.rmse(S=stft)
            feature_stats('rmse', f)
            
            f = librosa.feature.spectral_centroid(S=stft)
            feature_stats('spectral_centroid', f)
            f = librosa.feature.spectral_bandwidth(S=stft)
            feature_stats('spectral_bandwidth', f)
            f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
            feature_stats('spectral_contrast', f)
            f = librosa.feature.spectral_rolloff(S=stft)
            feature_stats('spectral_rolloff', f)
            
            mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
            del stft
            f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
            feature_stats('mfcc', f)
            
        except Exception as e:
            print('error')

        return np.array(features)
    

# fonction retourne numero de music a partir de path
def clean(path):
    path=path[-10:-4]
    for i in range(len(path)):
        if path[i]!='0':
            return path[i:]
    return path
            
#fonction retourne les path de tout les chansons
def data():
    path=r'C:\Users\HP\Desktop\fma_small\*'
    data_path = []
    data_num = []
    for filename in glob.glob(path):
        for music in glob.glob(filename+'\*.mp3'):
            data_path.append(music)
    return data_path 


#fonction retourn une dataframe des features de tout les chansons
def features():
    data_p=data()
    list_data=[]
    for a in range(len(data_p)):
        try :
            t=list(compute_feature(data_p[a]))
        except ValueError :
            pass   
        t.insert(0,clean(data_p[a]))
        list_data.append(t)

                
    print(pd.DataFrame(list_data))
    pd.DataFrame(list_data).to_csv(r"C:\Users\HP\Desktop\dataN.csv", mode='a', header=False)

if __name__=='__main__' : 
    features()
#fabriquer une fichier finale donneF qui contient le chanson ses features et et son genre
    te=pd.read_csv(r"C:\Users\HP\Desktop\dataN.csv", header=None,low_memory=False)
    ta=pd.read_csv(r"C:\Users\HP\Desktop\fma_metadata\tracks.csv", low_memory=False)
    te=te.iloc[:,1:]
    ta=ta.set_index('Unnamed: 0')
    ta=ta.iloc[2:,:]
    l=['track.7','set']
    ta=ta[ta['set.1']=='small']
    ta=ta[l]
    te=te.set_index(te.iloc[:,0])
    frames=[ta,te]
    result = ta.merge(te, on=ta.index, how = 'inner')
    result.iloc[:,1:].to_csv(r"C:\Users\HP\Desktop\donneF.csv")





