# ${1}: path to the input file ex: "./data/public.jsonl"
# ${2}: path to the output file ex:"./trainingResults2/summary_result.jsonl"

python3.9 summaryResult.py --input_file ${1} --output_file ${2} --num_beams 5
# example:
# python3.9 summaryResult.py --input_file "./data/public.jsonl" --output_file "./trainingResults2/summary_result.json" --num_beams 5

# training
# python3.9 run_summarization_no_trainer.py --train_file ${1} --validation_file ${2} --output_file ${3} --num_beams 5
# example:
# python3.9 run_summarization_no_trainer.py --train_file "./data/train.jsonl" --validation_file "./data/public.jsonl" --output_file "./summary_result.json"

