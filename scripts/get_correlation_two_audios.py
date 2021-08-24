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
    # return np.abs(np.sum(audio_1 * audio_2)) / (
    #     np.sqrt(np.sum(audio_1 ** 2)) * np.sqrt(np.sum(audio_2 ** 2))
    # )
    # return np.sum(audio_1 * audio_2) / (
    #     np.sqrt(np.sum(audio_1 ** 2)) * np.sqrt(np.sum(audio_2 ** 2))
    # )
    return np.sum(audio_1 * audio_2)


def find_correlation_audios_from_a_source(
    source_audio_path: str, target_audios_dir: str, target_audios_names: list = None
):
    """
    Find the correlation between audios from a source audio
    @param source_audio_path: source audio path
    @param target_audios_dir: target audios directory
    @param target_audios_names: target audios names. If given, we restrict the correlation to these audios only
    @return: correlation between audios from a source audio
    """
    source_audio, sr = librosa.load(source_audio_path, sr=None)
    correlations = []
    if target_audios_names is None:
        target_audios_names = sorted(os.listdir(target_audios_dir))

    for i, file_name_1 in enumerate(sorted(os.listdir(target_audios_dir))):
        if file_name_1.endswith(".wav"):
            file_name_1 = file_name_1.replace(".wav", "")
            if file_name_1 not in target_audios_names:
                print(
                    f"[Warn] {file_name_1} is not on the target_audios_names list. Ignore!"
                )
                continue
            target_audio_path = os.path.join(target_audios_dir, file_name_1 + ".wav")
            target_audio, sr = librosa.load(target_audio_path, sr=None)
            # plt.subplot(1, 2, 1)
            # librosa.display.waveplot(source_audio, sr=sr)
            # plt.subplot(1, 2, 2)
            # librosa.display.waveplot(target_audio, sr=sr)
            # plt.show()
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
    source_audio_dir2 = "/home/tynguyen/github_ws/sound-spaces/data/sounds/my_audios/insane_piano_2s.wav"

    # target_audios_dir = "/home/tynguyen/bags/sound_nerf_raw_data/audios_cellos_1s_river_flow_in_you_combined"
    target_audios_dir = (
        "/home/tynguyen/bags/sound_nerf_raw_data/audios_cello_0_insane_piano_39"
    )
    # We also find the correlation between the source audio and the target audios whose name are in the following list
    target_audios_names = [
        "nodeID_52_angleID_100",
        "nodeID_53_angleID_120",
        "nodeID_54_angleID_110",
        "nodeID_55_angleID_120",
        "nodeID_56_angleID_120",
        "nodeID_57_angleID_130",
    ]
    print(f"--------------------------------")
    print(f"\n[Info] Finding correlation between combined audios v.s. the cello audio")
    source1_corrs = find_correlation_audios_from_a_source(
        source_audio_dir1, target_audios_dir, target_audios_names
    )
    print(f"--------------------------------")
    print(
        f"\n[Info] Finding correlation between combined audios v.s. the  bach prelude audio"
    )
    source2_corrs = find_correlation_audios_from_a_source(
        source_audio_dir2, target_audios_dir, target_audios_names
    )

    print(f"--------------------------------")
    print(f"[Info] Cello correlations: \n{source1_corrs}")
    print(f"[Info] Bach correlations: \n{source2_corrs}")
    # Draw correlations
    X = np.arange(len(source1_corrs))
    plt.plot(X, source1_corrs, label="cello")
    plt.plot(X, source2_corrs, label="piano")
    plt.legend(["cello", "piano"])
    plt.show()
