"""
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
@Brief: Find the correlation between two audio signals
"""
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile


def find_two_audios_correlation_from_data(
    audio_1: np.ndarray, audio_2: np.ndarray
) -> float:
    """
    Find the correlation between two audio signals
    @param audio_1: audio signal 1
    @param audio_2: audio signal 2
    @return: correlation between two audio signals
    """
    len1 = len(audio_1)
    len2 = len(audio_2)
    if len1 > len2:
        audio_1 = audio_1[:len2]
    elif len1 < len2:
        audio_2 = audio_2[:len1]
    return np.abs(np.sum(audio_1 * audio_2)) / (
        np.sqrt(np.sum(audio_1 ** 2)) * np.sqrt(np.sum(audio_2 ** 2))
    )


def find_correlation_audios_from_a_source(
    source_audio_path: str, target_audios_dir: str
):
    """
    Find the correlation between audios from a source audio
    @param source_audio_path: source audio path
    @param target_audios_dir: target audios directory
    @return: correlation between audios from a source audio
    """
    source_audio, sr = librosa.load(source_audio_path, sr=None)
    correlations = []
    for i, file_name_1 in enumerate(os.listdir(target_audios_dir)):
        if file_name_1.endswith(".wav"):
            file_name_1 = file_name_1.replace(".wav", "")
            target_audio_path = os.path.join(target_audios_dir, file_name_1 + ".wav")
            target_audio, sr = librosa.load(target_audio_path, sr=None)
            plt.subplot(1, 2, 1)
            librosa.display.waveplot(source_audio, sr=sr)
            plt.subplot(1, 2, 2)
            librosa.display.waveplot(target_audio, sr=sr)
            plt.show()
            correlation = find_two_audios_correlation_from_data(
                source_audio, target_audio
            )
            print(
                f"[Info] {i} Processing file {file_name_1}, correlation {correlation}"
            )
            correlations.append(correlation)

    print(f"[Info] End!")
    print(f"[Info] Processed {i+1} audios in total!")
    return correlations


if __name__ == "__main__":
    source_audio_dir1 = (
        "/home/tynguyen/github_ws/sound-spaces/data/sounds/my_audios/cello_1s.wav"
    )
    source_audio_dir2 = "/home/tynguyen/github_ws/sound-spaces/data/sounds/my_audios/bach_prelude_fugue_N2_in_c_minor.wav"
    target_audios_dir = "/home/tynguyen/bags/sound_nerf_raw_data/audios_cellos_1s_bach_piano_1s_combined"
    print(f"--------------------------------")
    print(f"\n[Info] Finding correlation between combined audios v.s. the cello audio")
    source1_corrs = find_correlation_audios_from_a_source(
        source_audio_dir1, target_audios_dir
    )
    print(f"--------------------------------")
    print(
        f"\n[Info] Finding correlation between combined audios v.s. the  bach prelude audio"
    )
    source2_corrs = find_correlation_audios_from_a_source(
        source_audio_dir2, target_audios_dir
    )

    print(f"--------------------------------")
    print(f"[Info] Cello correlations: \n{source1_corrs}")
    print(f"[Info] Bach correlations: \n{source2_corrs}")
    # Draw correlations
    X = np.arange(len(source1_corrs))
    plt.plot(X, source1_corrs, label="cello")
    plt.plot(X, source2_corrs, label="bach")
    plt.legend(["cello", "bach"])
    plt.show()
