import os
import numpy as np
import pandas as pd
from funasr import AutoModel

model = AutoModel(model="iic/emotion2vec_base_finetuned") # Alternative: iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large and iic/emotion2vec_base_finetuned

def test_default_inference():
    wav_file = f"{model.model_path}/example/test.wav"
    rec_result = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
    print(rec_result)

output_path = './wav.scp'
audio_folder_path = './test/'


if __name__ == "__main__":
    try:
        rec_result = model.generate(output_path,
                                    output_dir="./outputs",
                                    granularity="utterance",
                                    extract_embedding=False)
        data = []
        for entry in rec_result:
            key = entry['key']
            labels = entry['labels']
            scores = entry['scores']

            # Round the scores to 3 decimal places
            rounded_scores = [round(score, 3) for score in scores]

            # Find the index of the maximum score
            max_score_index = scores.index(max(scores))

            # Get the corresponding label for the maximum score
            predict_label = labels[max_score_index]

            # Append a dictionary with the key, scores, and predict_label
            data.append({'key': key, **dict(zip(labels, rounded_scores)), 'predict_label': predict_label})

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data)

    except Exception as error:
        print(f"Something went error: {error}")