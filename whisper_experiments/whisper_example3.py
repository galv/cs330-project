import fire
import jiwer

import torch


import datasets
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from my_dataset import MyDataset

def main(manifest_json_path="/home/dgalvez/scratch/code/classes/cs330/project/speech-wikimedia/output_nfa_segment_split/part-00000-e10f693c-60c7-41b8-ad07-5ecf61d9d7fb-c000_with_output_file_paths.json"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v2"

    processor = AutoProcessor.from_pretrained(model_id)

    dataset = MyDataset(manifest_json_path, processor)
    data_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, num_workers=1)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    references = []
    hypotheses = []

    for input_features, ref_text in data_loader:
        assert input_features.shape[0] == 1
        input_features = input_features.cuda().half()
        predicted_ids = model.generate(
            input_features[0],
            prompt_ids=processor.get_prompt_ids("cat,fish,animal", return_tensors="pt"))
        transcription = processor.batch_decode(predicted_ids)
        print("Hypothesis:", transcription)
        print("Reference:", ref_text)

    print("GALVEZ:wer=", jiwer.wer(references, hypotheses))
    

if __name__ == "__main__":
    fire.Fire(main)
