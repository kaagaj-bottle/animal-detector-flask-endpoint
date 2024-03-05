import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor
from pathlib import Path
from load_model import auto_transforms
from pydub import AudioSegment
import librosa
import soundfile
import os
from typing import Dict, List, Tuple


default_mfcc_params = {
    "n_fft": 2048,
    "win_length": None,
    "n_mels": 256,
    "hop_length": 512,
    "htk": True,
    "norm_melspec": None,
    "n_mfcc": 256,
    "norm_mfcc": "ortho",
    "dct_type": 2,
}


def mfcc_transform(y: np.ndarray,
                   sr: int,
                   mfcc_params: Dict = default_mfcc_params
                   ) -> torch.Tensor:
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=mfcc_params['n_fft'],
        win_length=mfcc_params['win_length'],
        hop_length=mfcc_params['hop_length'],
        n_mels=mfcc_params['n_mels'],
        htk=mfcc_params['htk'],
        norm=mfcc_params['norm_melspec'],
    )
    mfcc_librosa = librosa.feature.mfcc(
        S=librosa.core.spectrum.power_to_db(melspec),
        n_mfcc=mfcc_params['n_mfcc'],
        dct_type=mfcc_params['dct_type'],
        norm=mfcc_params['norm_mfcc']
    )

    return torch.from_numpy(mfcc_librosa)


def expand_tensor_for_efficient_net(y) -> torch.Tensor:
    input_tensor_expanded = y.expand(3, -1, -1)
    resized_tensor = F.interpolate(input_tensor_expanded.unsqueeze(
        0), size=(224, 224), mode='bilinear', align_corners=False)

    resized_tensor = resized_tensor.squeeze()
    return resized_tensor

# preprocesses single image to tensor of size and transformation as per pre-trained model


def preprocess_single_image(img):
    img_as_tensor = pil_to_tensor(img)
    transformed_img = auto_transforms(img_as_tensor)
    return transformed_img

# takes input as blobs and outputs the input tensor for the model comprising of all the images


def handle_blobs(blobs):
    no_of_frames = len(blobs)
    input_tensor = torch.zeros(
        [no_of_frames, 3, 224, 224], dtype=torch.float32)
    idx = 0

    for key in blobs:
        try:
            img = Image.open(blobs[key].stream)
        except Exception as e:
            print('----------------------------------------------')
            print(e)
            print('----------------------------------------------')
            return input_tensor
        input_tensor[idx] = preprocess_single_image(img)
        idx += 1
    return input_tensor


def save_images(blobs):
    i = 0
    for key in blobs:
        img = Image.open(blobs[key].stream)
        try:
            img.save(f"img{i}.bmp")
            i += 1
        except:
            print(f"couldn't save the {i}th image")


def temporal_segmentation_single_file(file: Path, segment_duration: int = 5):

    y, sr = librosa.load(file, sr=None)

    duration = librosa.get_duration(y=y, sr=sr)
    no_of_segments = 0

    if (duration > segment_duration):
        no_of_segments = int(duration//segment_duration)
        segments = []
        for i in range(no_of_segments):

            start_time = i*segment_duration
            end_time = (i+1)*segment_duration
            segment = y[int(start_time)*sr:int(end_time)*sr]
            segments.append([segment, sr])
        return segments
    else:
        return []


def convert_single_file_to_wav(file: Path, output_path: Path):
    INPUT_FORMAT = "mp3"
    OUTPUT_FORMAT = "wav"
    try:
        sound = AudioSegment.from_file(file, format=INPUT_FORMAT)
    except:
        print(f"couldn't read the file: {file}")
    try:
        sound.export(output_path, format=OUTPUT_FORMAT)
    except:
        print(f"couldn't write the file: {file}")


def clean_dir(dir_path: Path):
    file_paths = [dir_path/Path(item) for item in os.listdir(dir_path)]
    [os.remove(item) for item in file_paths]


def preprocess_segments(segments: List):
    segments_len = len(segments)
    input_tensor = torch.zeros(
        [segments_len, 3, 224, 224], dtype=torch.float32)
    for i in range(segments_len):
        transformed = mfcc_transform(segments[i][0], segments[i][1],)
        input_tensor[i] = expand_tensor_for_efficient_net(transformed)
    return input_tensor


def save_load_audio(blob):

    audio_input_dir = Path("audio_input")
    audio_input_dir.mkdir(exist_ok=True, parents=True)
    clean_dir(audio_input_dir)

    file_path = audio_input_dir / Path("audio.mp3")
    output_path = Path(str(file_path).replace(".mp3", ".wav"))
    blob.save(file_path)
    convert_single_file_to_wav(file_path, output_path)

    segments = temporal_segmentation_single_file(output_path)
    preprocessed_segments = preprocess_segments(segments)
    return preprocessed_segments
