# bash whisper_experiments/run_title_prompt.sh 2>&1 > logs/run_title_prompt_no_append_previous_transcripts_to_prompt.log

# bash whisper_experiments/run_description_prompt.sh 2>&1 > logs/run_description_prompt_no_append_previous_transcripts_to_prompt.log

# bash whisper_experiments/run_nouns_and_proper_nouns_prompt.sh 2>&1 > logs/run_nouns_and_proper_nouns_prompt_no_append_previous_transcripts_to_prompt.log

# bash whisper_experiments/run_none_prompt.sh 2>&1 > logs/run_none_prompt_no_append_previous_transcripts_to_prompt.log

bash whisper_experiments/run_title_prompt.sh 2>&1 > logs/run_title_prompt_append_previous_transcripts_to_prompt.log &

bash whisper_experiments/run_description_prompt.sh 2>&1 > logs/run_description_prompt_append_previous_transcripts_to_prompt.log &

bash whisper_experiments/run_nouns_and_proper_nouns_prompt.sh 2>&1 > logs/run_nouns_and_proper_nouns_prompt_append_previous_transcripts_to_prompt.log &

bash whisper_experiments/run_none_prompt.sh 2>&1 > logs/run_none_prompt_append_previous_transcripts_to_prompt.log &

wait
