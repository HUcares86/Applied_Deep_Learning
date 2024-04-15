
# $1 -> seen / unseen
python multiLabel/inference.py --path data --model_path 3_model.ckpt --task $1 --output subgroup_$1_predict.csv