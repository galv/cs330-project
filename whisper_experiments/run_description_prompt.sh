echo "Description prompt"

# python create_prompts.py --prompt_type=description --output_file_json=output_prompts/description_prompt.json

# python whisper_experiments/whisper_example3.py --file_to_prompt_path=output_prompts/description_prompt.json --hypotheses_file=output_prompts/description_hyps.txt --references_file=output_prompts/description_refs.txt

python whisper_experiments/whisper_example3.py --file_to_prompt_path=output_prompts/description_prompt.json --hypotheses_file=output_prompts/description_append_previous_hyps.txt --references_file=output_prompts/description_append_previous_refs.txt --append_previous_transcripts_to_prompt=True
