import json

import fire
import jiwer

import torch
import tqdm

import datasets
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from my_dataset import MyDataset

def main(manifest_json_path="/home/dgalvez/scratch/code/classes/cs330/project/speech-wikimedia/output_nfa_segment_split/part-00000-e10f693c-60c7-41b8-ad07-5ecf61d9d7fb-c000_with_output_file_paths.json",
         file_to_prompt_path="TODO.json"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # model_id = "openai/whisper-large-v2" # small
    # model_id = "distil-whisper/distil-large-v2"
    model_id = "distil-whisper/distil-medium.en"

    processor = AutoProcessor.from_pretrained(model_id)

    dataset = MyDataset(manifest_json_path, processor)
    data_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, num_workers=1)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True, use_flash_attention_2=True, low_cpu_mem_usage=True)
    model.to(torch.float16)
    model.to(device)

    references = []
    hypotheses = []

    with open(file_to_prompt_path) as fh:
        file_to_prompt_dictionary = json.load(fh)

    for i, (input_features, ref_text, audio_filepath) in enumerate(tqdm.tqdm(data_loader)):
        if i == 2:
            torch.cuda.cudart().cudaProfilerStart()
        if i == 3:
            torch.cuda.cudart().cudaProfilerStop()
        assert input_features.shape[0] == 1
        assert len(audio_filepath) == 1
        prompt = file_to_prompt_dictionary[audio_filepath[0]]
        input_features = input_features.cuda().half()
        predicted_ids = model.generate(
            input_features[0],
            prompt_ids=processor.get_prompt_ids(prompt, return_tensors="pt"))
        transcription = processor.batch_decode(predicted_ids)

        new_hypotheses = [t[t.find("<|notimestamps|> ") + len("<|notimestamps|> "):t.find("<|endoftext|>")] for t in transcription]

        # print("GALVEZ:", new_hypotheses)

        hypotheses.extend(new_hypotheses)
        references.extend(ref_text)

    print("GALVEZ:wer=", jiwer.wer(references, hypotheses))
    

if __name__ == "__main__":
    fire.Fire(main)
