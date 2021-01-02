#import libreries:
import numpy as np
import librosa
import time
import os, glob
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import webbrowser


#extract_feature to extract the mfcc, chroma and mel features from a sound file
def extract_features(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X= sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate

        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])

        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))

        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result=np.hstack((result, chroma))

        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result=np.hstack((result, mel))
            return result



#define dictionary to hold numbers and the emotion available in the dataset
emotions={
    '01':'calm',
    '02':'neutral',
    '03':'happy',
     '04':'sad',
     '05':'angry',
     '06':'fearful',
     '07':'disgust',
     '08':'surprised'
}
observed_emotions=['calm','neutral', 'angry', 'happy']

#load the data and extract fettures
def load_data(test_size=0.25):
    x,y=[],[]
    for file in glob.glob("speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_features(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#split dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.3)

#get shape of training and test dataset
#print((x_train.shape[0],x_test.shape[0]))

#get the number of features extraction:
#print(f'Features extracted:{x_train.shape[1]}')

#initialize the MLP classifier:
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300), learning_rate='adaptive', max_iter=500)

#Train the model
model.fit(x_train,y_train)

#pridict for the set
y_pred=model.predict(x_test)

#calculate the accuracy of our model:
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy of our model is: {:.2f}%".format(accuracy*100))
#print(y_pred)

#load the input file:
file="speech-emotion-recognition-ravdess-data\\Actor_17\\03-01-01-01-01-01-17.wav"
feature=extract_features(file, mfcc=True, chroma=True, mel=True)

#split d dataset
x_train, x_test, y_train, y_test=load_data(test_size=0.3)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

#Initialize the Multi Layer Perceptron Classifier:
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300), learning_rate='adaptive', max_iter=500)

#train the model:
model.fit(x_train,y_train)

#predict for the test_set:
y_pred=model.predict(x_test)
y_pre=model.predict([feature])
print(y_pre)
time.sleep(2)


if y_pre[0]=="calm":
    webbrowser.open('https://media.giphy.com/media/l1J9NRpOeS7i54xnW/giphy.gif')
elif y_pre[0]=="neutral":
    webbrowser.open('https://media.giphy.com/media/39lYbuIEDqiDHAD0KT/giphy.gif')
elif y_pre[0] == "happy":
    webbrowser.open('https://media.giphy.com/media/XbxZ41fWLeRECPsGIJ/giphy.gif')
elif y_pre[0] == "sad":
    webbrowser.open('https://media.giphy.com/media/l1KVaj5UcbHwrBMqI/giphy.gif')
elif y_pre[0] == "angry":
    webbrowser.open('https://media.giphy.com/media/l1J9u3TZfpmeDLkD6/giphy.gif')
elif y_pre[0] == "fearful":
    webbrowser.open('https://media.giphy.com/media/fQDSxFtuo7jkdWZem2/giphy.gif')
elif y_pre[0] == "disgust":
    webbrowser.open('https://media.giphy.com/media/bupsZiBKn7vAk/giphy.gif')
elif y_pre[0] == "surprised":
    webbrowser.open('https://media.giphy.com/media/ksEi1DprJUroG66Z99/giphy.gif')
else:
    webbrowser.open('https://media.giphy.com/media/KD7tIgAbqZCDAir9hZ/giphy.gif')


