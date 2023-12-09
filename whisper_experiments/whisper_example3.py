import json

import fire
import jiwer

import re
import torch
import tqdm

import time

import datasets
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from my_dataset import MyDataset

def main(manifest_json_path="/home/dgalvez/scratch/code/classes/cs330/project/speech-wikimedia/output_nfa_segment_split/part-00000-e10f693c-60c7-41b8-ad07-5ecf61d9d7fb-c000_with_output_file_paths.json",
         file_to_prompt_path="TODO.json",
         append_previous_transcripts_to_prompt=False,
         hypotheses_file="refs.txt",
         references_file="hyps.txt"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v2" # small
    # model_id = "distil-whisper/distil-large-v2"
    # model_id = "distil-whisper/distil-medium.en"

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


    current_sample_id = None
    current_previous_transcript = None

    start_time = time.time()
    for i, (sample_id, input_features, ref_text, audio_filepath) in enumerate(data_loader): # tqdm.tqdm
        # print("GALVEZ4:", i)
        if i == 2:
            torch.cuda.cudart().cudaProfilerStart()
        if i == 3:
            torch.cuda.cudart().cudaProfilerStop()
        assert input_features.shape[0] == 1
        assert len(audio_filepath) == 1


        if sample_id != current_sample_id:
            current_previous_transcript = ""
            current_sample_id = sample_id
        prompt = file_to_prompt_dictionary[audio_filepath[0]]
        # print("GALVEZ:,max_target_positions=", model.config.max_target_positions)
        
        # Need to be smarter about the prompt here. Use model.max_target_positions
        if append_previous_transcripts_to_prompt:
            if prompt != "":
                full_prompt = prompt + ". " + current_previous_transcript
            else:
                full_prompt = current_previous_transcript
        else:
            full_prompt = prompt
        input_features = input_features.cuda().half()
        # print("GALVEZ: full prompt=", full_prompt)
        prompt_ids = processor.get_prompt_ids(full_prompt, return_tensors="pt")
        # print("GALVEZ: prompt size", prompt_ids.size())
        # if (append_previous_transcripts_to_prompt and
        #     len(prompt_ids) >= model.config.max_target_positions):
        #     # for i in range(len(current_transcript_list) - 1, -1, -1):
        #     for i in range(len(current_transcript_list)):
        #         if prompt != "":
        #             full_prompt = prompt + ". " + " ".join(current_transcript_list[i:])
        #         else:
        #             full_prompt = " ".join(current_transcript_list[i:])
        #         prompt_ids = processor.get_prompt_ids(full_prompt, return_tensors="pt")
        #         if len(prompt_ids) < model.config.max_target_positions:
        #             break
        # print("GALVEZ:truncated prompt=", full_prompt)
        # print("GALVEZ:truncated prompt size=", prompt_ids.size())
        predicted_ids = model.generate(
            input_features[0],
            prompt_ids=prompt_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

        # print("GALVEZ5:", transcription[0])


        # Need a better way to do this
        # new_hypotheses = [t.replace("<|notimestamps|>", "").replace("<|endoftext|>", "").replace("<|startofprev|>", "").replace("<|startoftranscript|>", "") for t in transcription]
        # Need a better way to do this
        # new_hypotheses = [t[t.find("<|notimestamps|>") + len("<|notimestamps|>"):t.find("<|endoftext|>")] for t in transcription]
        new_hypotheses = []
        for t in transcription:
            t = t[t.find("<|startoftranscript|>") + len("<|startoftranscript|>"):]
            t = re.sub(r"<\|.+?\|>", "", t)
            t = t.lstrip(" ")
            new_hypotheses.append(t)

        # print("AFTER:(", new_hypotheses[0], ")")

        assert len(new_hypotheses) == 1

        current_previous_transcript = new_hypotheses[0]

        # print("GALVEZ:", new_hypotheses)

        hypotheses.extend(new_hypotheses)
        references.extend(ref_text)

    end_time = time.time()
    print("GALVEZ:wer=", jiwer.wer(references, hypotheses))
    print("GALVEZ: total time=", end_time - start_time)
    
    with open(hypotheses_file, "w") as fh:
        for line in hypotheses:
            fh.write(f"{line}\n")

    with open(references_file, "w") as fh:
        for line in references:
            fh.write(line)


if __name__ == "__main__":
    fire.Fire(main)
