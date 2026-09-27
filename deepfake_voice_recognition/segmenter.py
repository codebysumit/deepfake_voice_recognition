"""
segmenter.py
------------
Audio dataset segmenter and preprocessor for deepfake voice recognition.

Splits multi-length or long audio files into standardized, fixed-duration
chunks (e.g., 4 or 6 seconds) resampled to a target sample rate (e.g., 16kHz)
and padded with silence if needed, while preserving class and language
directory structures.

Usage:
    # As a CLI tool:
    python -m deepfake_voice_recognition.segmenter --source-root ./data/raw --output-root ./data/segmented

    # As a Python module:
    from deepfake_voice_recognition.segmenter import AudioSegmenter, process_dataset
    segmenter = AudioSegmenter(segment_seconds=4.0, target_sr=16000)
    result = segmenter.process_dataset(source_root="raw_data", output_root="segmented_data")
"""

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from .config import cfg
except ImportError:
    try:
        from config import cfg
    except ImportError:
        cfg = None


DEFAULT_AUDIO_EXTENSIONS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")
DEFAULT_SPOOF_KEYWORDS: Tuple[str, ...] = ("spoof", "fake", "clone", "cloned", "deepfake", "ai_generated")
DEFAULT_GENUINE_KEYWORDS: Tuple[str, ...] = ("genuine", "real", "bonafide", "human", "original")


class AudioSegmenter:
    """
    Handles scanning, loading, resampling, slicing, padding, and saving audio
    files into fixed-length segments suitable for deepfake voice detection training.
    """

    def __init__(
        self,
        segment_seconds: float = 4.0,
        target_sr: int = 16000,
        audio_extensions: Sequence[str] = DEFAULT_AUDIO_EXTENSIONS,
        spoof_keywords: Sequence[str] = DEFAULT_SPOOF_KEYWORDS,
        genuine_keywords: Sequence[str] = DEFAULT_GENUINE_KEYWORDS,
        mono: bool = True,
        pad_mode: str = "constant",
    ):
        self.segment_seconds = float(segment_seconds)
        self.target_sr = int(target_sr)
        self.audio_extensions = tuple(audio_extensions)
        self.spoof_keywords = tuple(spoof_keywords)
        self.genuine_keywords = tuple(genuine_keywords)
        self.mono = mono
        self.pad_mode = pad_mode

        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be strictly positive.")
        if self.target_sr <= 0:
            raise ValueError("target_sr must be strictly positive.")

    @property
    def chunk_samples(self) -> int:
        return int(self.target_sr * self.segment_seconds)

    def find_audio_files(self, source_root: str) -> List[Dict[str, str]]:
        """
        Recursively discovers audio files in source_root and extracts class/language info.

        Expected folder layouts supported:
            1. SOURCE_ROOT/<class>/<language>/<filename>
            2. SOURCE_ROOT/<class>/<filename>
        """
        if not os.path.exists(source_root):
            raise FileNotFoundError(f"Source root directory not found: {source_root}")

        found_items = []

        for ext in self.audio_extensions:
            pattern = os.path.join(source_root, "**", f"*{ext}")
            for file_path in glob.glob(pattern, recursive=True):
                norm_path = os.path.normpath(file_path)
                rel_path = os.path.relpath(norm_path, source_root)
                parts = rel_path.split(os.sep)

                # Determine class and language
                label = None
                language = "default"

                for part in parts[:-1]:
                    part_lower = part.lower()
                    if any(k in part_lower for k in self.spoof_keywords):
                        label = "fake"
                        break
                    elif any(k in part_lower for k in self.genuine_keywords):
                        label = "real"
                        break

                if label is None:
                    # Check the immediate parent
                    parent = os.path.basename(os.path.dirname(norm_path)).lower()
                    if any(k in parent for k in self.spoof_keywords):
                        label = "fake"
                    elif any(k in parent for k in self.genuine_keywords):
                        label = "real"
                    else:
                        print(f"[Warning] Skipping file with unclassified folder: {file_path}")
                        continue

                # Determine language if nested (e.g., real/Bengali/audio.wav or fake/English/audio.wav)
                if len(parts) >= 3:
                    # Parent of file is language, grandparent is class
                    language = parts[-2]
                elif len(parts) == 2:
                    # real/audio.wav -> language is default
                    language = "all"

                found_items.append({
                    "path": norm_path,
                    "language": language,
                    "label": label,
                    "rel_path": rel_path,
                })

        return sorted(found_items, key=lambda x: x["path"])

    def split_into_chunks(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Splits a 1D audio array into equal-length chunks.
        Pads the trailing chunk with silence if shorter than chunk_samples.
        """
        chunk_len = self.chunk_samples
        total_samples = len(audio)

        if total_samples == 0:
            return [np.zeros(chunk_len, dtype=np.float32)]

        chunks = []
        for start in range(0, total_samples, chunk_len):
            chunk = audio[start:start + chunk_len]
            if len(chunk) < chunk_len:
                pad_amount = chunk_len - len(chunk)
                chunk = np.pad(chunk, (0, pad_amount), mode=self.pad_mode)
            chunks.append(chunk.astype(np.float32))

        return chunks

    def process_file(
        self,
        file_path: str,
        output_dir: str,
        base_id: Optional[str] = None,
    ) -> List[str]:
        """
        Loads, resamples, chunks, and writes segments for a single audio file.
        Returns the list of generated output file paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        audio, sr = librosa.load(file_path, sr=self.target_sr, mono=self.mono)
        chunks = self.split_into_chunks(audio)

        file_stem = os.path.splitext(os.path.basename(file_path))[0]
        if base_id is None:
            if "_" in file_stem:
                file_id, rest = file_stem.split("_", 1)
            else:
                file_id, rest = file_stem, ""
        else:
            file_id, rest = base_id, file_stem

        saved_paths = []
        for index, chunk in enumerate(chunks, start=1):
            segment_no = f"{index:02d}"
            if rest:
                out_name = f"{file_id}_{segment_no}_{rest}.wav"
            else:
                out_name = f"{file_id}_{segment_no}.wav"

            out_path = os.path.join(output_dir, out_name)
            sf.write(out_path, chunk, self.target_sr, subtype="PCM_16")
            saved_paths.append(out_path)

        return saved_paths

    def process_dataset(
        self,
        source_root: str,
        output_root: str,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Processes all discovered audio files in source_root and saves segmented
        audio files under output_root organized as:
            output_root/real/<language>/<chunks>.wav
            output_root/fake/<language>/<chunks>.wav
        """
        audio_files = self.find_audio_files(source_root)
        print(f"Found {len(audio_files)} audio files under: {source_root}")

        if not audio_files:
            print("[Warning] No matching audio files found to segment.")
            return {
                "total_files": 0,
                "total_chunks": 0,
                "failed_files": [],
                "output_root": output_root,
            }

        os.makedirs(output_root, exist_ok=True)
        total_chunks = 0
        failed_files = []

        iterator = tqdm(audio_files, desc="Segmenting audio files") if show_progress else audio_files

        for item in iterator:
            file_path = item["path"]
            language = item["language"]
            label = item["label"]

            out_dir = os.path.join(output_root, label, language)

            try:
                created = self.process_file(file_path, out_dir)
                total_chunks += len(created)
            except Exception as exc:
                print(f"[Error] Failed to process {file_path}: {exc}")
                failed_files.append({"path": file_path, "error": str(exc)})

        summary = {
            "total_files": len(audio_files),
            "processed_files": len(audio_files) - len(failed_files),
            "failed_files": failed_files,
            "total_chunks": total_chunks,
            "segment_seconds": self.segment_seconds,
            "target_sr": self.target_sr,
            "source_root": source_root,
            "output_root": output_root,
        }

        print("\n" + "=" * 55)
        print("Audio Segmentation Complete")
        print(f"Source folder    : {source_root}")
        print(f"Output folder    : {output_root}")
        print(f"Segment length   : {self.segment_seconds}s ({self.chunk_samples} samples)")
        print(f"Sample rate      : {self.target_sr} Hz")
        print(f"Files processed  : {summary['processed_files']} / {summary['total_files']}")
        print(f"Files failed     : {len(failed_files)}")
        print(f"Total chunks     : {total_chunks}")
        print("=" * 55)

        return summary


def find_audio_files(source_root: str) -> List[Dict[str, str]]:
    """Convenience module-level function for finding files."""
    segmenter = AudioSegmenter()
    return segmenter.find_audio_files(source_root)


def split_into_chunks(audio: np.ndarray, sr: int = 16000, segment_seconds: float = 4.0) -> List[np.ndarray]:
    """Convenience module-level function for chunking a 1D audio array."""
    segmenter = AudioSegmenter(segment_seconds=segment_seconds, target_sr=sr)
    return segmenter.split_into_chunks(audio)


def process_dataset(
    source_root: Optional[str] = None,
    output_root: Optional[str] = None,
    segment_seconds: float = 4.0,
    target_sr: int = 16000,
) -> Dict[str, Any]:
    """Convenience function to run full dataset segmentation."""
    if source_root is None:
        source_root = cfg.raw_dataset_root if (cfg and hasattr(cfg, "raw_dataset_root")) else "./data/raw"
    if output_root is None:
        output_root = cfg.dataset_root if cfg else "./data/segmented"

    segmenter = AudioSegmenter(segment_seconds=segment_seconds, target_sr=target_sr)
    return segmenter.process_dataset(source_root=source_root, output_root=output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="AudioSegmenter",
        description="Split raw audio files into fixed-length, standardized chunks for voice deepfake recognition.",
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default=getattr(cfg, "raw_dataset_root", "./data/raw"),
        help="Path to source directory with raw real/fake audio.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=getattr(cfg, "dataset_root", "./data/segmented"),
        help="Path to directory where segmented audio dataset will be saved.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=getattr(cfg, "max_seconds", 4.0),
        help="Target length of each audio segment in seconds (default: 4.0).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=getattr(cfg, "sample_rate", 16000),
        help="Target sample rate in Hz (default: 16000).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    segmenter = AudioSegmenter(
        segment_seconds=args.segment_seconds,
        target_sr=args.sample_rate,
    )
    segmenter.process_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
