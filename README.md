# VoiceOfTheNation

This project is made by Group 11 of course CS-671 at IIT Mandi.

We have trained deep learning models to classify an audio into one of the 9 Indian languages, namely Bengali,Tamil,Telugu,Gujarati,Marathi, Malayalam, Kannada, Urdu and Hindi.

The dataset we have used for the project is Audio Dataset with 10 Indian Languages, which is openly available on Kaggle.

# Basic pipeline that we used is :-
Data Preparation -> Data Preprocessing -> Model Building ->  Training -> Evaluation

-> Data Preparation = In this step, the files are renamed as "language_name'_'original_file_name'.mp3 and has been stored into a single folder instead of class folders.

-> Data Preprocessing = In this process following tasks are performed on the prepared dataset:
--Silence Trimming
--Normalisation
--Resampling
--Feature Extraction ( using librosa for cnn based models and log mel spectrogram for whisper based models).

-> Model Building = Two different models have been trained on completely different architectures- one is the cnn based approach and other is the whisper based approach.

-> Training= cnns are fed with images of mel spectrogram and whisper is fed with log mel spectrogram and the models are trained.

-> Evaluation= The following methods are used for evaluating the models:
--Accuracy
--Class wise and overall f1 scores
--confusion matrix

*****************************************************************************

Respective graphs for training+ validation loss vs epochs and embeddings visualization using t-SNE/PCA after each stage have also been plotted in both the models.
 
We also have made gradio apps for both the models. The cnn model is saved as .h5 file and whisper model is saved as .pt file. After that, two apps are made for both using their saved file path. The app uses microphone to capture real time audio data, and the use it for prediction of language using the respective trained models. 

*****************************************************************************

# FUTURE WORK:-
->Instead of using mel spectrogram and related features, the models can be trained on the raw input given as a vector. This will eliminate any kind of computational noise or compression, which will help to improve efficiency of the models.
->Training on more languages
->Efficient language detection on shorter clips.

This read.me file is not ai-generated, it is written by us ourselves :)
Thank you