import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
RANDOM_SEED = 7
import librosa
import librosa.display
from librosa.core import load
import parselmouth
from utils.feature_extraction_utils import *

# region feature extraction 
def get_stats_from_feature(feature_input):
    feature_mean,feature_median = np.mean(feature_input.T, axis=0),np.median(feature_input.T, axis=0)
    feature_std  = np.std(feature_input.T, axis=0)
    feature_p10, feature_p90  = np.percentile(feature_input.T, 10, axis=0), np.percentile(feature_input.T, 90, axis=0)
    return np.concatenate((feature_mean,feature_median,feature_std, feature_p10, feature_p90), axis=0)

def calc_feature_all(sample_file_path):
    sample_rate = 16000
    X, _ = librosa.load(sample_file_path, res_type='kaiser_fast',duration=4.5,sr=16000,offset=3.0)
    mfccs_60 = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=60)
    feature_mfccs_60_stats = get_stats_from_feature(mfccs_60)
    stft = np.abs(librosa.stft(X))
    feature_chroma_stft_stats = get_stats_from_feature(librosa.feature.chroma_stft(S=stft, sr=sample_rate))
    feature_mel_32_stats = get_stats_from_feature(librosa.feature.melspectrogram(y=X, sr=sample_rate,
                                                         n_fft=2048, hop_length=512,
                                                         n_mels=32, fmax=8000))
    feature_mel_64_stats = get_stats_from_feature(librosa.feature.melspectrogram(y=X, sr=sample_rate,
                                                         n_fft=2048, hop_length=512,
                                                         n_mels=64, fmax=8000))
    feature_mel_128_stats = get_stats_from_feature(librosa.feature.melspectrogram(y=X, sr=sample_rate,
                                                         n_fft=2048, hop_length=512,
                                                         n_mels=128, fmax=8000))
    
    feature_zcr_stats = get_stats_from_feature(librosa.feature.zero_crossing_rate(y=X))
    feature_rms_stats = get_stats_from_feature(librosa.feature.rms(y=X))
    
    features  = np.concatenate((feature_mfccs_60_stats,
                                    feature_chroma_stft_stats,
                                    feature_mel_32_stats,
                                    feature_mel_64_stats,
                                    feature_mel_128_stats,
                                    feature_zcr_stats,
                                    feature_rms_stats
                                  ), axis=0)
    prefixes = {'mfcc': 60, 'chroma': 12, 'mel32': 32, 'mel64': 64,'mel128': 128, 'zcr': 1, 'rms': 1}
    column_names = []
    for prefix, num_features in prefixes.items():
        for prefix_stats in ['mean','median','std','p10','p90']:
            if num_features  > 1: 
                column_names.extend([f'{prefix}_{prefix_stats}_{i}' for i in range(1, num_features + 1)])
            else:
                column_names.extend([f'{prefix}_{prefix_stats}'])
    assert len(column_names) == 5*(60+12+32+64+128+2) 
    
    feature_part1= {}
    for key, value in zip(column_names, features):
        feature_part1[key] = value

    sound = parselmouth.Sound(values=X,sampling_frequency=sample_rate,start_time=0)
    intensity_attributes = get_intensity_attributes(sound)[0]
    pitch_attributes = get_pitch_attributes(sound)[0]
    hnr_attributes = get_harmonics_to_noise_ratio_attributes(sound)[0]
    local_jitter = get_local_jitter(sound)
    local_shimmer = get_local_shimmer(sound)
    spectrum_attributes = get_spectrum_attributes(sound)[0]
    formant_attributes = get_formant_attributes(sound)[0]
    expanded_intensity_attributes = {f"Intensity_{key}": value for key, value in intensity_attributes.items()}
    expanded_pitch_attributes = {f"Pitch_{key}": value for key, value in pitch_attributes.items()}
    expanded_hnr_attributes = {f"HNR_{key}": value for key, value in hnr_attributes.items()}
    expanded_spectrum_attributes = {f"Spectrum_{key}": value for key, value in spectrum_attributes.items()}
    expanded_formant_attributes = {f"Formant_{key}": value for key, value in formant_attributes.items()}
    feature_prosody = {
            **expanded_intensity_attributes,  # Unpack expanded intensity attributes
            **expanded_pitch_attributes,  # Unpack expanded pitch attributes
            **expanded_hnr_attributes,  # Unpack expanded HNR attributes
            "Local Jitter": local_jitter,
            "Local Shimmer": local_shimmer,
            **expanded_spectrum_attributes,  # Unpack expanded spectrum attributes
            **expanded_formant_attributes,  # Unpack expanded formant attribute
        }
    feature_combined = {**feature_part1,**feature_prosody}
    return feature_combined
# endregion


# region data label processing
def get_duration(file_path):
    try:
        y, sr = librosa.load(file_path)
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        yt, index = librosa.effects.trim(y,top_db=20)
        duration_seconds_trim = librosa.get_duration(y=yt, sr=sr)
        return round(duration_seconds,2), round(duration_seconds_trim,2)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

# Dictionary to map emotional categories to sentiment values
sentiment_mapping = {
    "Anger": -1,
    "Angry": -1,
    "Disgust": -1,
    "Fear": -1,
    "Fearful": -1,
    "Happy": 1,
    "Happiness": 1,
    "Surprised": 1,
    "Surprise": 1,
    "Neutral": 0,
    "Neutrality": 0,
    "Calm": 0,
    "Calmness": 0,
    "Sad": -1,
    "Sadness": -1
}

# Function to map emotional category to sentiment value
def map_to_sentiment(emotional_category):
    return sentiment_mapping.get(emotional_category, 0)

# endregion

# region feature name selection
def generate_selected_features_by_type(feature_column_names, feat_cat, stats, number=1):
    selected_result = []
    for name in feature_column_names:
        if feat_cat+ "_" + stats in name:
            selected_result.append(name)
    if number < len(selected_result):
        selected_result = selected_result[:number]
    return selected_result

# endregion