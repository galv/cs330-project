from collections import namedtuple
import glob
import json
import os
import pathlib

from torch.utils.data import Dataset
import torchaudio

DataChunk = namedtuple("DataChunk", "id, audio_filepath, text, start, end")

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
                        self.data.append(DataChunk(id, dictionary["audio_filepath"], text, start, start + duration))

        self.previous_audio_id = None
        self.previous_waveform = None

    def __getitem__(self, i):
        data = self.data[i]

        if data.id == self.previous_audio_id:
            waveform = self.previous_waveform
        else:
            waveform, sample_rate = torchaudio.load(data.audio_filepath)
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
            self.previous_audio_id = data.id
            self.previous_waveform = waveform

        waveform = waveform[:, round(data.start * 16_000):round(data.end * 16_000)]
        input_features = self.processor(waveform[0, :], sampling_rate=16_000, return_tensors="pt").input_features
        return data.id, input_features, data.text, data.audio_filepath

    def __len__(self):
        return len(self.data)
