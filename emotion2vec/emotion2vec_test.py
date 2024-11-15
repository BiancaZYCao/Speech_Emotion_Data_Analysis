import os
import numpy as np
import pandas as pd
from funasr import AutoModel

# model = AutoModel(model="iic/emotion2vec_base_finetuned") # Alternative: iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large and iic/emotion2vec_base_finetuned

# def test_default_inference():
#     wav_file = f"{model.model_path}/example/test.wav"
#     rec_result = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
#     print(rec_result)

output_path = './wav.scp'
audio_folder_path = './test/'
metadata = pd.read_csv("./metadata.csv")


def prepare_test_set_scp_file():
    with open(output_path, 'w') as wav_scp:
        # Loop through all files in the directory
        for file_name in os.listdir(audio_folder_path):
            if file_name.endswith('.wav'):
                # Extract wav_id (filename without extension)
                wav_id = os.path.splitext(file_name)[0]
                # Construct the full path
                wav_path = os.path.join(audio_folder_path, file_name)
                # Write to wav.scp in Kaldi format
                wav_scp.write(f"{wav_id} \t {wav_path}\n")

    print(f"wav.scp generated at: {output_path}")

# def inference_from_scp_file():
#     model.generate(output_path, output_dir="./outputs",
#                    granularity="utterance", extract_embedding=False)

if __name__ == "__main__":
    try:
        # test_default_inference()
        prepare_test_set_scp_file()
        # inference_from_scp_file()
    except Exception as error:
        print(f"Something went error: {error}")