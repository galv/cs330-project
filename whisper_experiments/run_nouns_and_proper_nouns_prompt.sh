echo "Nouns and proper nouns prompt"

# python create_prompts.py --prompt_type=nouns_and_proper_nouns --output_file_json=output_prompts/nouns_and_proper_nouns_prompt.json

# python whisper_experiments/whisper_example3.py --file_to_prompt_path=output_prompts/nouns_and_proper_nouns_prompt.json --hypotheses_file=output_prompts/nouns_and_proper_nouns_hyps.txt --references_file=output_prompts/nouns_and_proper_nouns_refs.txt

python whisper_experiments/whisper_example3.py --file_to_prompt_path=output_prompts/nouns_and_proper_nouns_prompt.json --hypotheses_file=output_prompts/nouns_and_proper_nouns_append_hyps.txt --references_file=output_prompts/nouns_and_proper_nouns_append_refs.txt --append_previous_transcripts_to_prompt=True
