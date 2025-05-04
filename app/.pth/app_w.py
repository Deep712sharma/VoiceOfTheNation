import gradio as gr
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn

# Label mapping
label_map = {
    0: "bengali", 1: "gujarati", 2: "hindi", 3: "kannada", 4: "malayalam",
    5: "marathi", 6: "urdu", 7: "punjabi", 8: "tamil", 9: "telugu"
}

# Define the classifier class
class WhisperClassifier(nn.Module):
    def __init__(self, whisper_model, num_classes=10):
        super().__init__()
        self.encoder = whisper_model.encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(whisper_model.config.d_model, num_classes)

    def forward(self, input_features):
        with torch.no_grad():
            encoder_outputs = self.encoder(input_features=input_features)
        x = encoder_outputs.last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(2)
        return self.classifier(x)

# Load model and processor
def load_model(model_path="/home/dl/G11/App/whisper_classifier.pt", processor_path="processor", device="cpu"):
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-base")

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    whisper_model = get_peft_model(whisper_model, peft_config)

    model = WhisperClassifier(whisper_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, processor

# Inference function
def predict_language(audio):
    waveform, sr = torchaudio.load(audio)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    inputs = processor(waveform[0], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        logits = model(input_features)
        pred = torch.argmax(logits, dim=1).item()
    return f"Predicted Language: {label_map[pred]}"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model, processor = load_model(device=device)

# Gradio Interface
interface = gr.Interface(
    fn=predict_language,
    inputs=gr.Audio(sources="microphone", type="filepath", label="Speak or Upload Audio"),
    outputs=gr.Text(label="Predicted Language"),
    title="Real-time Language Identification with Whisper",
    description="Upload or record a short audio clip. The model will identify the language spoken."
)

interface.launch()
