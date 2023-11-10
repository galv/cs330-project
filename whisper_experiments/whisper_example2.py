import torch

import datasets
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v2"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)

# model.config.forced_decoder_ids

model.to(device)

import ipdb; ipdb.set_trace()

ds = load_dataset("common_voice", "fr", split="test", streaming=True)
ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
input_speech = next(iter(ds))["audio"]["array"]
# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "fr", task = "transcribe")
input_features = processor(input_speech, return_tensors="pt").input_features 
predicted_ids = model.generate(input_features, prompt_ids=processor.get_prompt_ids("cat,fish,animal", return_tensors="pt"))
transcription = processor.batch_decode(predicted_ids)

# prompt_ids

print(transcription)


