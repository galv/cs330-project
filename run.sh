python third_party/NeMo/tools/nemo_forced_aligner/align.py \
pretrained_name="stt_en_fastconformer_hybrid_large_pc" \
manifest_filepath=/home/dgalvez/scratch/code/classes/cs330/project/speech-wikimedia/nemo_manifest_jsonl_single/part-00000-e10f693c-60c7-41b8-ad07-5ecf61d9d7fb-c000.json \
output_dir=output_nfa_segment_split/ \
audio_filepath_parts_in_utt_id=1 \
batch_size=1 \
additional_segment_grouping_separator="'<segment_split>'" \
save_output_file_formats=["ctm"] \
transcribe_device="cuda" \
viterbi_device="cuda"
# should viterbi device be cpu?
