from collections import namedtuple
import glob
import json
import os
import pathlib

from torch.utils.data import Dataset
import torchaudio

DataChunk = namedtuple("DataChunk", "audio_filepath, text, start, end")

class MyDataset(Dataset):
    def __init__(self, manifest_json_path, processor):
        self.processor = processor
        self.data = []
        with open(manifest_json_path, "rt") as fh:
            for line in fh:
                dictionary = json.loads(line)
                with open(dictionary["segments_level_ctm_filepath"], "rt") as ctm_fh:
                    for ctm_line in ctm_fh:
                        id, _, start, duration, text = ctm_line.split(" ")
                        start = float(start)
                        duration = float(duration)
                        text = text.replace("<space>", " ")
                    self.data.append(DataChunk(dictionary["audio_filepath"], text, start, start + duration))

    def __getitem__(self, i):
        data = self.data[i]
        waveform, sample_rate = torchaudio.load(data.audio_filepath)
        waveform = waveform[:, round(data.start * sample_rate):round(data.end * sample_rate)]
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        input_features = self.processor(waveform[0, :], sampling_rate=16_000, return_tensors="pt").input_features
        return input_features, data.text

    def __len__(self):
        return len(self.data)
