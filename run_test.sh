#!/bin/bash

INPUT_FILE=$1
OUTPUT_DIR="temp"
mkdir $OUTPUT_DIR
python feature_dict_to_pickle.py --features $INPUT_FILE --output_dir "$OUTPUT_DIR/URLs_Testing/" --dataset_name test --label phish
mv "$OUTPUT_DIR/URLs_Testing/vectorizer_test.pkl" "$OUTPUT_DIR/URLs_Testing/vectorizer.pkl"
mv "$OUTPUT_DIR/URLs_Testing/X_train_processed_test.pkl" "$OUTPUT_DIR/URLs_Testing/X_test.pkl"
mv "$OUTPUT_DIR/URLs_Testing/y_train_test.pkl" "$OUTPUT_DIR/URLs_Testing/y_test.pkl"
rm "$OUTPUT_DIR/URLs_Testing/X_train_unprocessed_test.pkl"

output=$(python main.py --output_input_dir $OUTPUT_DIR --ignore_confirmation 2>&1 >/dev/null)
echo "$output" | grep Classifiers.py:521
