import gradio as gr
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load model
model = tf.keras.models.load_model("/home/dl/G11/App/audio_language_classifier (2).h5")

# Recreate label encoder manually (use same order as during training)
language_labels = ['tamil', 'malayalam', 'gujarati', 'hindi', 'telugu', 'kannada', 'bengali', 'odia', 'urdu']
lbl = LabelEncoder()
lbl.fit(language_labels)

def extract_features_from_audio(audio_path, sr=22050):
    try:
        audio_data, _ = librosa.load(audio_path, sr=sr)
        audio_data = librosa.effects.trim(audio_data, top_db=20)[0]
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        scaled_feature = np.mean(mfcc.T, axis=0)
        return scaled_feature
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {e}")

def predict_language_from_mic(audio_path):
    try:
        features = extract_features_from_audio(audio_path)
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        label = lbl.inverse_transform([predicted_class])[0]
        return f"Predicted Language: {label}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=predict_language_from_mic,
    inputs=gr.Audio(sources="microphone", type="filepath", label="Speak Now"),
    outputs="text",
    title="Language Identifier",
    description="Speak into the microphone to detect the language from your voice using a CNN model trained on MFCC features."
)

if __name__ == "__main__":
    interface.launch()
