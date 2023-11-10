python third_party/NeMo/tools/nemo_forced_aligner/align.py \
pretrained_name="stt_en_fastconformer_hybrid_large_pc" \
manifest_filepath=/home/galv/code/alignment/speech-wikimedia/nemo_manifest_jsonl_single/part-00000-c3768aba-1288-4d7f-a9fc-512250aae219-c000.json \
output_dir=output_nfa/ \
audio_filepath_parts_in_utt_id=1 \
batch_size=1 \
save_output_file_formats=["ctm"] \
transcribe_device="cuda" \
viterbi_device="cpu"
# should viterbi device be cpu?
