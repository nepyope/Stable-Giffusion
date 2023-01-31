import whisper
import torch
import time
model_fp32 = whisper.load_model(
    name="base",
    device="cpu")

#quantized_model = torch.quantization.quantize_dynamic(
#    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
#)
result = whisper.transcribe(model_fp32, '7d00d539-ffb2-4db7-9453-065ef5510f81.mp4') 
print(result['text'])
