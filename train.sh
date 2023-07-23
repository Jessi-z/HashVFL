set -e

HASH_LENGTH = 1
DEFENSE = False

python3 --dataset_name criteo --encode_length ${HASH_LENGTH} --defense ${DEFENSE} --model_path "dataset/sample_data.txt" train.py 
