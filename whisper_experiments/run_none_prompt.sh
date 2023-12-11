echo "None prompt"

python create_prompts.py --prompt_type=none --output_file_json=output_prompts/none_prompt.json

python whisper_experiments/whisper_example3.py --file_to_prompt_path=output_prompts/none_prompt.json --hypotheses_file=output_prompts/none_hyps.txt --references_file=output_prompts/none_refs.txt

python whisper_experiments/whisper_example3.py --file_to_prompt_path=output_prompts/none_prompt.json --hypotheses_file=output_prompts/none_append_hyps.txt --references_file=output_prompts/none_append_refs.txt --append_previous_transcripts_to_prompt=True
