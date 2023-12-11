echo "Title prompt"

python whisper_experiments/whisper_example3.py --file_to_prompt_path=output_prompts/title_prompt.json --hypotheses_file=output_prompts/title_hyps.txt --references_file=output_prompts/title_refs.txt

python whisper_experiments/whisper_example3.py --file_to_prompt_path=output_prompts/title_prompt.json --hypotheses_file=output_prompts/title_apprend_previous_hyps.txt --references_file=output_prompts/title_append_previous_refs.txt --append_previous_transcripts_to_prompt=True
