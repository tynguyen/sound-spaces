"""
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
@Brief: Add two audio together
"""
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile


def add_two_audios_from_files(
    audio_file_1: str, audio_file_2: str, output_file_path: str
) -> None:
    """
    Add two audio files together
    @param audio_file_1: First audio file
    @param audio_file_2: Second audio file
    @param output_file_path: Path of output file
    """
    audio_1, sr_1 = librosa.load(audio_file_1, sr=None)
    audio_2, sr_2 = librosa.load(audio_file_2, sr=None)
    plt.subplot(1, 2, 1)
    librosa.display.waveplot(audio_1, sr_1)
    plt.subplot(1, 2, 2)
    librosa.display.waveplot(audio_2, sr_2)
    plt.show()
    assert sr_1 == sr_2, "[Error] Sampling rate of audio files are not the same!"
    assert len(audio_1) == len(
        audio_2
    ), "[Error] Length of audio files are not the same!"
    output_audio = audio_1 + audio_2
    soundfile.write(output_file_path, output_audio, sr_1)


def add_audios_in_pairs(audios_1_dir: str, audios_2_dir: str, output_dir: str):
    """
    Add two audio files together, pair by pair
    @param audios_1_dir: directory of audios 1
    @param audios_2_dir: directory of audios 2
    @param output_dir: directory of output
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, file_name_1 in enumerate(os.listdir(audios_1_dir)):
        if file_name_1.endswith(".wav"):
            print(f"[Info] {i} Processing file {file_name_1}")
            assert file_name_1 in os.listdir(
                audios_2_dir
            ), "[Error] File {file_name_1} does not appear on {audios_2_dir}"
            file_name_1 = file_name_1.replace(".wav", "")
            output_file_name = file_name_1 + ".wav"
            output_file_path = os.path.join(output_dir, output_file_name)
            add_two_audios_from_files(
                os.path.join(audios_1_dir, file_name_1 + ".wav"),
                os.path.join(audios_2_dir, file_name_1 + ".wav"),
                output_file_path,
            )
    print(f"[Info] End!")
    print(f"[Info] Processed {i+1} audios in total!")


if __name__ == "__main__":
    audio1_files_dir = (
        "/home/tynguyen/bags/sound_nerf_raw_data/audios_river_flow_in_you"
    )
    audio2_files_dir = "/home/tynguyen/bags/sound_nerf_raw_data/audios_cellos_1s"
    add_audios_in_pairs(
        audio1_files_dir,
        audio2_files_dir,
        "/home/tynguyen/bags/sound_nerf_raw_data/audios_cellos_1s_river_flow_in_you_combined",
    )
