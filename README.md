# Using the instructions and other findings

1. Installing NeMo on a fresh python environment in the server was not possible using pip. Eventually the solution was to clone the repo to use the `align.py` script.

2. The `make_manifest_sentence_segments.py` uses two parameters: tgt_manifest and src_data_dir. The second one has some very specific requirements: it must contain a flattened directory with all the audios and transcriptions kept there.
Given that People's Speech already comes in NeMo format (we already have the manifest both for the validation and the test set) this script can't be used directly, as a method to convert the JSON manifest to .srt files would be required. 

3. Omitting the use of the previous script has some additional problems, as segmenting the data is a requirement to use the aligner.

4. In order to get the `align.py` script tested, both the validation_000000.json file as well as the first TAR file were downloaded from HuggingFace. The validation json contains 921 filepaths at the `[training_data][name]` key and after uncompressing the TAR file a total of 921 files are stored at the same path.

5. The following code was run

``` 
python3 /home/rafael/supervised_peoples_speech/instructions_for_aligning_peoples_speech_test_set/NeMo/tools/nemo_forced_aligner/align.py \
pretrained_name="stt_en_fastconformer_hybrid_large_pc" \
manifest_filepath=validation_000000.json \
output_dir=/home/rafael/supervised_peoples_speech/> \
audio_filepath_parts_in_utt_id=1 \
additional_ctm_grouping_separator="'<segment_split>'" \
batch_size=1 \
save_output_file_formats=["ctm"] \
transcribe_device="cuda" \
viterbi_device="cuda"
```

After running the code, an error specifying that the additional_ctm_grouping_separator could not be overwritten popped up. I gave it another try without that parameter and got the following error:

    RuntimeError: At least one line in cfg.manifest_filepath does not contain an 'audio_filepath' entry. All lines must contain an 'audio_filepath' entry.
