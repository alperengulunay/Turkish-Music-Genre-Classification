import matplotlib.pyplot as plt
import numpy as np

import librosa as lb
import librosa.display

SR = 22050
N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64     
count = 150
song_list = [
                'Arabesk',
                'Caz',
                'Elektronik',
                'Punk',
                'Tasavvuf'
              ]
genre = 'tasavvuf'
file = r'-'
PATH = r'C:\Songs'
offset = 60 #time to start


for i in range(1, count+1):
    try:
        path = PATH +'\\' + file +'\\' + genre + '_{:03}'.format(i) + '.mp3'
            
        data, sr = lb.load(path, mono=True, offset=offset, duration=2)
            
        melspec = lb.feature.melspectrogram(y=data, sr=44100, hop_length = HOP_LENGTH, n_fft = N_FFT, n_mels = N_MELS)
            
        fig, ax = plt.subplots()
            
        S_dB = lb.power_to_db(melspec**2, ref=np.max)
            
        plt.figure(figsize= (224,224))
            
        img = lb.display.specshow(S_dB, sr=44100, ax=ax)
        plt.show()
    except:
        print('error on' , path)