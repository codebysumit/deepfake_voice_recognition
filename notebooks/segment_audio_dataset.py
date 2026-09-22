"""
Audio Dataset Segmenter
========================

What this script does:
1. Reads audio files from a source folder.
   The source folder must look like this:

       SOURCE_ROOT/
           spoof/   (or "fake")
               Bengali/
                   file1.wav
                   file2.mp3
               English/
                   file3.wav
           genuine/ (or "real")
               Bengali/
                   ...
               English/
                   ...

2. Splits each audio file into fixed length chunks.
   The chunk length (in seconds) is set by SEGMENT_SECONDS below.

3. If the last chunk is shorter than SEGMENT_SECONDS, it adds silence
   (zero padding) at the end, so every chunk has the same length.

4. Saves each chunk as a .wav file, in the same language/class
   structure, inside OUTPUT_ROOT.

This follows the same style as the reference notebook:
one config block at the top, tqdm progress bars, and clear print
summaries after each step.
"""

import os
import glob

import numpy as np
import soundfile as sf
import librosa
from tqdm.auto import tqdm

# ------------------------------------------------------------
# Config (edit these values only, the rest of the code does not
# need to change)
# ------------------------------------------------------------

# Folder that holds your raw audio, sorted by language and class
SOURCE_ROOT = "/content/raw-audio-dataset"

# Folder where the fixed-length chunks will be saved
OUTPUT_ROOT = "/content/segmented-audio-dataset"

# Chunk length in seconds. Change this to 4, 6, or any value you need.
SEGMENT_SECONDS = 4

# Sample rate to use for every output file (Hz).
# All audio gets resampled to this rate, so every chunk has the
# exact same number of samples.
TARGET_SR = 16000

# File types to look for inside each class folder
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

# Folder-name keywords used to detect the class label.
# Add more words here if your folders use different names.
SPOOF_KEYWORDS = ("spoof", "fake", "clone", "cloned", "deepfake")
GENUINE_KEYWORDS = ("genuine", "real", "bonafide")


# ------------------------------------------------------------
# Step 1: find every audio file and work out its language + class
# ------------------------------------------------------------

def find_audio_files(source_root):
    """
    Walks through source_root and returns a list of dicts:
        {"path": full file path,
         "language": folder name for language,
         "label": "spoof" or "genuine"}

    The class folder (spoof/genuine) must be the direct parent of
    the language folder, and the language folder must be the direct
    parent of the audio file. That is:
    SOURCE_ROOT/<class>/<language>/<file>
    """
    found = []

    for ext in AUDIO_EXTENSIONS:
        pattern = os.path.join(source_root, "**", f"*{ext}")
        for file_path in glob.glob(pattern, recursive=True):
            language_dir = os.path.basename(os.path.dirname(file_path))
            class_dir = os.path.basename(
                os.path.dirname(os.path.dirname(file_path))
            )

            class_lower = class_dir.lower()
            if any(word in class_lower for word in SPOOF_KEYWORDS):
                label = "spoof"
            elif any(word in class_lower for word in GENUINE_KEYWORDS):
                label = "genuine"
            else:
                # Folder name did not match a known keyword, skip it
                # and print a warning so nothing is silently lost.
                print(f"Skipping (unknown class folder): {file_path}")
                continue

            found.append({
                "path": file_path,
                "language": language_dir,
                "label": label,
            })

    return found


# ------------------------------------------------------------
# Step 2: split one audio array into fixed-length, zero-padded chunks
# ------------------------------------------------------------

def split_into_chunks(audio, sr, segment_seconds):
    """
    Splits a 1-D audio array into a list of equal-length chunks.
    The final chunk is zero-padded at the end if it is shorter
    than segment_seconds.
    """
    chunk_len = int(sr * segment_seconds)

    if chunk_len <= 0:
        raise ValueError("segment_seconds * sr must be greater than 0")

    chunks = []
    total_samples = len(audio)

    for start in range(0, total_samples, chunk_len):
        chunk = audio[start:start + chunk_len]

        if len(chunk) < chunk_len:
            pad_amount = chunk_len - len(chunk)
            chunk = np.pad(chunk, (0, pad_amount), mode="constant")

        chunks.append(chunk)

    return chunks


# ------------------------------------------------------------
# Step 3: process every file and save the chunks
# ------------------------------------------------------------

def process_dataset(source_root=SOURCE_ROOT,
                     output_root=OUTPUT_ROOT,
                     segment_seconds=SEGMENT_SECONDS,
                     target_sr=TARGET_SR):

    audio_files = find_audio_files(source_root)
    print(f"Found {len(audio_files)} audio files under: {source_root}")

    total_chunks = 0
    failed_files = []

    for item in tqdm(audio_files, desc="Splitting audio files"):
        file_path = item["path"]
        language = item["language"]
        label = item["label"]

        try:
            # sr=target_sr resamples on load, so every file ends up
            # at the same sample rate before we split it
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        except Exception as error:
            print(f"Could not read file: {file_path} ({error})")
            failed_files.append(file_path)
            continue

        chunks = split_into_chunks(audio, sr, segment_seconds)

        out_dir = os.path.join(output_root, label, language)
        os.makedirs(out_dir, exist_ok=True)

        # Original file name looks like: <file_id>_<tts_model_name>_<voice_gender>
        # Example: "101_Gemini 3.1 Flash TTS" (before the extension)
        # We only need the file_id, which is the part before the first "_"
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        if "_" in base_name:
            file_id, rest_of_name = base_name.split("_", 1)
        else:
            file_id, rest_of_name = base_name, ""

        for index, chunk in enumerate(chunks, start=1):
            segment_no = f"{index:02d}"

            if rest_of_name:
                out_name = f"{file_id}_{segment_no}_{rest_of_name}.wav"
            else:
                out_name = f"{file_id}_{segment_no}.wav"

            out_path = os.path.join(out_dir, out_name)
            sf.write(out_path, chunk, sr)

        total_chunks += len(chunks)

    print("\n" + "=" * 50)
    print("Audio segmentation completed")
    print(f"Source folder    : {source_root}")
    print(f"Output folder    : {output_root}")
    print(f"Segment length   : {segment_seconds} seconds")
    print(f"Sample rate      : {target_sr} Hz")
    print(f"Files processed  : {len(audio_files) - len(failed_files)}")
    print(f"Files failed     : {len(failed_files)}")
    print(f"Total chunks made: {total_chunks}")
    print("=" * 50)

    return {
        "total_files": len(audio_files),
        "total_chunks": total_chunks,
        "failed_files": failed_files,
    }


if __name__ == "__main__":
    process_dataset()
