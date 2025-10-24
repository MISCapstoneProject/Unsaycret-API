#!/usr/bin/env python3
"""
Utility script to build a two-speaker speech separation dataset from the TCC-300
corpus. The script discovers speakers, plans clip quotas, assembles 4-second
mixtures with optional augmentation, and emits WAV files and metadata CSVs for
train/valid/test splits.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import multiprocessing
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

CLIP_DURATION_SECONDS = 4.0
SINGLE_SPEAKER_RATIO = 0.55
TWO_SPEAKER_RATIO = 0.45
OVERLAP_BUCKET_ORDER = ["0", "(0,20]", "(20,50]", "(50,100]"]
DEFAULT_OVERLAP_WEIGHTS = [0.25, 0.25, 0.30, 0.20]
BALANCED_OVERLAP_WEIGHTS = [0.25, 0.35, 0.30, 0.10]
OVERLAP_BUCKETS = {
    "0": (0.0, 0.0),
    "(0,20]": (0.01, 0.20),
    "(20,50]": (0.21, 0.50),
    "(50,100]": (0.51, 0.99),
}
COVERAGE_OPTIONS = (0.25, 0.5, 0.75, 1.0)
PLACEMENT_OPTIONS = ("front", "center", "back", "random")
GENDER_COMBOS = ("mm", "mf", "ff")
SPLIT_CONFIG = (("train", 0.8), ("valid", 0.1), ("test", 0.1))
DEFAULT_PAD_DB_RANGE = (-60.0, -50.0)
TARGET_SOURCE_RMS_RANGE = (0.05, 0.1)
RIR_PROBABILITY = 0.5
BG_NOISE_PROBABILITY = 0.5
EPS = 1e-12


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Speaker:
    speaker_id: str
    gender: str
    institute: str
    wavs: List[Path] = field(default_factory=list)


@dataclass
class ClipSpec:
    split: str
    global_index: int
    clip_id: str
    num_speakers: int
    coverage_ratio: float
    placement: str
    overlap_label: str
    overlap_range: Tuple[float, float]
    gender_combo: str
    primary_gender: str
    secondary_gender: Optional[str]
    rng_seed: int


@dataclass
class ClipAssignment:
    spec: ClipSpec
    speaker1: Speaker
    speaker2: Optional[Speaker]


@dataclass
class AssetManager:
    target_sr: int
    noise_files: List[Path]
    rir_files: List[Path]
    noise_cache: Dict[Path, np.ndarray] = field(default_factory=dict)
    rir_cache: Dict[Path, np.ndarray] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_noise(self, path: Path) -> np.ndarray:
        with self.lock:
            cached = self.noise_cache.get(path)
            if cached is not None:
                return cached
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            audio = audio.astype(np.float32)
            self.noise_cache[path] = audio
            return audio

    def get_rir(self, path: Path) -> np.ndarray:
        with self.lock:
            cached = self.rir_cache.get(path)
            if cached is not None:
                return cached
            rir, sr = sf.read(path)
            if rir.ndim > 1:
                rir = np.mean(rir, axis=1)
            if sr != self.target_sr:
                rir = librosa.resample(rir, orig_sr=sr, target_sr=self.target_sr)
            rir = rir.astype(np.float32)
            self.rir_cache[path] = rir
            return rir


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def db_to_amplitude(db: float) -> float:
    return 10.0 ** (db / 20.0)


def distribute_counts(total: int, weights: Sequence[float]) -> List[int]:
    if total <= 0:
        return [0 for _ in weights]
    total_weight = float(sum(weights))
    raw = [total * (w / total_weight) for w in weights]
    counts = [int(math.floor(val)) for val in raw]
    remainder = total - sum(counts)
    fractions = [val - math.floor(val) for val in raw]
    order = np.argsort(fractions)[::-1]
    for idx in order[:remainder]:
        counts[int(idx)] += 1
    return counts


def placement_to_start(placement: str, segment_samples: int, canvas_samples: int, rng: np.random.Generator) -> int:
    max_start = canvas_samples - segment_samples
    if max_start <= 0:
        return 0
    if placement == "front":
        return 0
    if placement == "center":
        return max(0, (canvas_samples - segment_samples) // 2)
    if placement == "back":
        return max_start
    if placement == "random":
        return int(rng.integers(0, max_start + 1))
    raise ValueError(f"Unknown placement {placement}")


def feasible_overlap_fraction(segment_samples: int, canvas_samples: int) -> float:
    overlap = max(0, 2 * segment_samples - canvas_samples)
    return min(1.0, overlap / segment_samples if segment_samples > 0 else 0.0)


def categorize_overlap(ratio: float) -> str:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio <= 1e-3:
        return "0"
    if ratio <= 0.20 + 1e-6:
        return "(0,20]"
    if ratio <= 0.50 + 1e-6:
        return "(20,50]"
    return "(50,100]"


def relative_to(value: Path, base: Path) -> str:
    return str(value.relative_to(base).as_posix())


def choose_two_distinct(items: Sequence[Speaker], rng: np.random.Generator) -> Tuple[Speaker, Speaker]:
    if not items:
        raise ValueError("Empty speaker list.")
    if len(items) == 1:
        return items[0], items[0]
    idx1 = int(rng.integers(0, len(items)))
    idx2 = idx1
    attempts = 0
    while idx2 == idx1 and attempts < 10:
        idx2 = int(rng.integers(0, len(items)))
        attempts += 1
    if idx1 == idx2:
        idx2 = (idx1 + 1) % len(items)
    return items[idx1], items[idx2]


def sample_overlap_ratio(bucket: str, rng: np.random.Generator) -> float:
    """Draw an overlap ratio within the bucket boundaries."""
    if bucket == "0":
        return 0.0
    if bucket == "(0,20]":
        return float(rng.uniform(0.01, 0.20))
    if bucket == "(20,50]":
        return float(rng.uniform(0.21, 0.50))
    if bucket == "(50,100]":
        return float(rng.uniform(0.51, 0.99))
    raise ValueError(f"Unknown overlap bucket: {bucket}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(out_root: Path) -> logging.Logger:
    ensure_dir(out_root / "logs")
    log_path = out_root / "logs" / "build.log"
    logger = logging.getLogger("dataset_builder")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. Output -> %s", log_path)
    return logger


# ---------------------------------------------------------------------------
# Speaker discovery and splitting
# ---------------------------------------------------------------------------


def discover_speakers(tcc_root: Path, logger: logging.Logger) -> List[Speaker]:
    if not tcc_root.exists():
        raise FileNotFoundError(f"TCC-300 root not found: {tcc_root}")
    speakers: Dict[str, Speaker] = {}
    for institute_dir in sorted(tcc_root.glob("*")):
        if not institute_dir.is_dir():
            continue
        institute = institute_dir.name
        for gender_dir in ("male", "female"):
            gender_path = institute_dir / gender_dir
            if not gender_path.exists():
                logger.warning("Missing gender folder: %s", gender_path)
                continue
            for speaker_dir in sorted(gender_path.iterdir()):
                if not speaker_dir.is_dir():
                    continue
                wavs = sorted(p for p in speaker_dir.glob("*.wav"))
                if not wavs:
                    logger.warning("Speaker has no WAV files: %s", speaker_dir)
                    continue
                speaker_id = f"{institute}_{gender_dir}_{speaker_dir.name}"
                speakers[speaker_id] = Speaker(
                    speaker_id=speaker_id,
                    gender=gender_dir,
                    institute=institute,
                    wavs=wavs,
                )
    if not speakers:
        raise RuntimeError("No speakers discovered under TCC-300 root.")
    logger.info("Discovered %d speakers across institutes.", len(speakers))
    return list(speakers.values())


def split_speakers(
    speakers: Sequence[Speaker],
    rng: np.random.Generator,
    logger: logging.Logger,
) -> Dict[str, List[Speaker]]:
    groups: Dict[Tuple[str, str], List[Speaker]] = defaultdict(list)
    for speaker in speakers:
        groups[(speaker.institute, speaker.gender)].append(speaker)

    split_map: Dict[str, List[Speaker]] = {"train": [], "valid": [], "test": []}
    for key, group in groups.items():
        rng.shuffle(group)
        counts = distribute_counts(len(group), [ratio for _, ratio in SPLIT_CONFIG])
        offset = 0
        for idx, (split, _) in enumerate(SPLIT_CONFIG):
            n = counts[idx]
            split_map[split].extend(group[offset : offset + n])
            offset += n

    for split, collection in split_map.items():
        logger.info("Split %s -> %d speakers", split, len(collection))
    return split_map


# ---------------------------------------------------------------------------
# Planning quotas and clip specs
# ---------------------------------------------------------------------------


def plan_clip_specs(
    total_clips: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    overlap_weights: Sequence[float],
) -> Dict[str, List[ClipSpec]]:
    split_counts = {}
    split_weights = [ratio for _, ratio in SPLIT_CONFIG]
    counts_per_split = distribute_counts(total_clips, split_weights)
    for idx, (split, _) in enumerate(SPLIT_CONFIG):
        split_counts[split] = counts_per_split[idx]

    clip_specs: Dict[str, List[ClipSpec]] = {split: [] for split, _ in SPLIT_CONFIG}
    global_index = 0
    for split, split_total in split_counts.items():
        if split_total == 0:
            continue
        single_count, two_count = distribute_counts(split_total, [SINGLE_SPEAKER_RATIO, TWO_SPEAKER_RATIO])

        coverage_counts = distribute_counts(split_total, [1, 1, 1, 1])
        coverage_list: List[float] = []
        for cov, count in zip(COVERAGE_OPTIONS, coverage_counts):
            coverage_list.extend([cov] * count)
        rng.shuffle(coverage_list)

        placement_list = [random.choice(PLACEMENT_OPTIONS) for _ in range(split_total)]

        single_gender_counts = distribute_counts(single_count, [1, 1])
        single_genders = ["male"] * single_gender_counts[0] + ["female"] * single_gender_counts[1]
        rng.shuffle(single_genders)

        combo_counts = distribute_counts(two_count, [1, 1, 1])
        combo_labels = (
            ["mm"] * combo_counts[0]
            + ["mf"] * combo_counts[1]
            + ["ff"] * combo_counts[2]
        )
        rng.shuffle(combo_labels)

        overlap_counts = distribute_counts(two_count, overlap_weights)
        overlap_labels = (
            ["0"] * overlap_counts[0]
            + ["(0,20]"] * overlap_counts[1]
            + ["(20,50]"] * overlap_counts[2]
            + ["(50,100]"] * overlap_counts[3]
        )
        rng.shuffle(overlap_labels)

        specs: List[ClipSpec] = []
        coverage_idx = 0
        placement_idx = 0

        for i in range(single_count):
            coverage = coverage_list[coverage_idx]
            placement = placement_list[placement_idx]
            coverage_idx += 1
            placement_idx += 1
            gender = single_genders[i]
            global_index += 1
            specs.append(
                ClipSpec(
                    split=split,
                    global_index=global_index,
                    clip_id=f"{global_index:06d}",
                    num_speakers=1,
                    coverage_ratio=coverage,
                    placement=placement,
                    overlap_label="NA",
                    overlap_range=(0.0, 0.0),
                    gender_combo="single",
                    primary_gender=gender,
                    secondary_gender=None,
                    rng_seed=rng.integers(0, 2**32 - 1, dtype=np.uint32).item(),
                )
            )

        for i in range(two_count):
            coverage = coverage_list[coverage_idx]
            placement = placement_list[placement_idx]
            coverage_idx += 1
            placement_idx += 1
            combo = combo_labels[i]
            overlap_label = overlap_labels[i]
            overlap_range = OVERLAP_BUCKETS[overlap_label]
            if combo == "mm":
                primary_gender = "male"
                secondary_gender = "male"
            elif combo == "ff":
                primary_gender = "female"
                secondary_gender = "female"
            else:
                primary_gender = "male"
                secondary_gender = "female"

            global_index += 1
            specs.append(
                ClipSpec(
                    split=split,
                    global_index=global_index,
                    clip_id=f"{global_index:06d}",
                    num_speakers=2,
                    coverage_ratio=coverage,
                    placement=placement,
                    overlap_label=overlap_label,
                    overlap_range=overlap_range,
                    gender_combo=combo,
                    primary_gender=primary_gender,
                    secondary_gender=secondary_gender,
                    rng_seed=rng.integers(0, 2**32 - 1, dtype=np.uint32).item(),
                )
            )

        rng.shuffle(specs)
        clip_specs[split] = specs
        logger.info(
            "Planned %d clips for split=%s (single=%d, two=%d).",
            len(specs),
            split,
            single_count,
            two_count,
        )
    return clip_specs


def assign_speakers_to_specs(
    clip_specs: Dict[str, List[ClipSpec]],
    split_speakers: Dict[str, List[Speaker]],
    rng: np.random.Generator,
    logger: logging.Logger,
) -> Dict[str, List[ClipAssignment]]:
    assignments: Dict[str, List[ClipAssignment]] = {}
    for split, specs in clip_specs.items():
        speakers = split_speakers.get(split, [])
        gender_map: Dict[str, List[Speaker]] = {"male": [], "female": []}
        for speaker in speakers:
            gender_map[speaker.gender].append(speaker)

        for gender, items in gender_map.items():
            if not items:
                logger.warning("Split %s has no speakers with gender=%s.", split, gender)

        split_assignments: List[ClipAssignment] = []
        for spec in specs:
            rng_local = np.random.default_rng(spec.rng_seed)
            if spec.num_speakers == 1:
                gender = spec.primary_gender
                candidates = gender_map.get(gender) or speakers
                if not candidates:
                    raise RuntimeError(f"No speakers available for split={split}")
                speaker = rng_local.choice(candidates)
                split_assignments.append(ClipAssignment(spec=spec, speaker1=speaker, speaker2=None))
            else:
                if spec.gender_combo == "mf":
                    male_candidates = gender_map.get("male") or speakers
                    female_candidates = gender_map.get("female") or speakers
                    if not male_candidates or not female_candidates:
                        raise RuntimeError(f"Insufficient speakers for combo mf in split {split}")
                    male_speaker = rng_local.choice(male_candidates)
                    female_speaker = rng_local.choice(female_candidates)
                    split_assignments.append(
                        ClipAssignment(
                            spec=spec,
                            speaker1=male_speaker,
                            speaker2=female_speaker,
                        )
                    )
                elif spec.gender_combo in ("mm", "ff"):
                    gender = "male" if spec.gender_combo == "mm" else "female"
                    candidates = gender_map.get(gender) or speakers
                    if not candidates:
                        raise RuntimeError(f"Insufficient speakers for combo {spec.gender_combo} in split {split}")
                    spk1, spk2 = choose_two_distinct(candidates, rng_local)
                    split_assignments.append(
                        ClipAssignment(
                            spec=spec,
                            speaker1=spk1,
                            speaker2=spk2,
                        )
                    )
                else:
                    raise ValueError(f"Unknown gender combo {spec.gender_combo}")
        assignments[split] = split_assignments
        logger.info("Assigned speakers for %d clips in split %s.", len(split_assignments), split)
    return assignments


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def sample_audio_segment(
    audio: np.ndarray,
    segment_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(audio) <= segment_samples:
        if len(audio) == segment_samples:
            return audio.copy()
    start_max = max(0, len(audio) - segment_samples)
    start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
    return audio[start : start + segment_samples].copy()


def extract_speech_segment(
    wav_path: Path,
    target_sr: int,
    segment_samples: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    energy_threshold_db: float = -40.0,
) -> np.ndarray:
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = audio.astype(np.float32)
    if len(audio) < segment_samples:
        logger.debug("Audio shorter than requested segment, will pad later. File=%s", wav_path)

    frame_length = min(2048, len(audio))
    hop_length = max(256, frame_length // 4)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length, center=True).flatten()
    if rms.size == 0:
        return sample_audio_segment(audio, segment_samples, rng)
    rms_db = librosa.power_to_db(rms**2 + EPS, ref=np.max(rms**2 + EPS))
    voiced_mask = rms_db > energy_threshold_db

    intervals: List[Tuple[int, int]] = []
    start_idx: Optional[int] = None
    for idx, voiced in enumerate(voiced_mask):
        if voiced and start_idx is None:
            start_idx = idx
        elif not voiced and start_idx is not None:
            intervals.append((start_idx, idx))
            start_idx = None
    if start_idx is not None:
        intervals.append((start_idx, len(voiced_mask)))

    candidates: List[Tuple[int, int]] = []
    for start, end in intervals:
        start_sample = librosa.frames_to_samples(start, hop_length=hop_length)
        end_sample = librosa.frames_to_samples(end, hop_length=hop_length)
        candidates.append((start_sample, end_sample))

    viable = [iv for iv in candidates if iv[1] - iv[0] >= segment_samples]
    if viable:
        chosen = viable[int(rng.integers(0, len(viable)))]
        start = chosen[0]
        end = start + segment_samples
        return audio[start:end].copy()

    if candidates:
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        seg = audio[chosen[0]:chosen[1]].copy()
    else:
        seg = sample_audio_segment(audio, segment_samples, rng)

    if len(seg) < segment_samples:
        seg = np.pad(seg, (0, segment_samples - len(seg)), mode="edge")
    elif len(seg) > segment_samples:
        seg = seg[:segment_samples]
    return seg


def generate_low_level_noise(
    length: int,
    target_dbfs: float,
    rng: np.random.Generator,
    asset_manager: AssetManager,
    use_external: bool = True,
) -> Tuple[np.ndarray, str]:
    noise_id = "PROC_NOISE"
    if use_external and asset_manager.noise_files:
        noise_path = rng.choice(asset_manager.noise_files)
        source = asset_manager.get_noise(noise_path)
        noise_id = noise_path.name
        segment = sample_audio_segment(source, length, rng)
    else:
        segment = rng.standard_normal(length).astype(np.float32)

    rms = np.sqrt(np.mean(segment**2) + EPS)
    target_rms = db_to_amplitude(target_dbfs)
    if rms < EPS:
        scale = 0.0
    else:
        scale = target_rms / rms
    segment = segment * scale
    return segment.astype(np.float32), noise_id


def maybe_apply_rir(
    track: np.ndarray,
    rng: np.random.Generator,
    asset_manager: AssetManager,
) -> Tuple[np.ndarray, str]:
    if not asset_manager.rir_files:
        return track, "NONE"
    if rng.random() > RIR_PROBABILITY:
        return track, "NONE"
    rir_path = rng.choice(asset_manager.rir_files)
    rir = asset_manager.get_rir(rir_path)
    convolved = fftconvolve(track, rir, mode="full")[: len(track)]
    return convolved.astype(np.float32), rir_path.name


def maybe_apply_background_noise(
    mix: np.ndarray,
    rng: np.random.Generator,
    asset_manager: AssetManager,
) -> Tuple[np.ndarray, str]:
    if not asset_manager.noise_files:
        return mix, "NONE"
    if rng.random() > BG_NOISE_PROBABILITY:
        return mix, "NONE"
    noise_path = rng.choice(asset_manager.noise_files)
    noise_audio = asset_manager.get_noise(noise_path)
    segment = sample_audio_segment(noise_audio, len(mix), rng)
    snr_db = rng.uniform(0.0, 8.0)
    mix_rms = np.sqrt(np.mean(mix**2) + EPS)
    noise_rms = np.sqrt(np.mean(segment**2) + EPS)
    if noise_rms < EPS or mix_rms < EPS:
        return mix, "NONE"
    desired_noise_rms = mix_rms / db_to_amplitude(snr_db)
    segment = segment * (desired_noise_rms / noise_rms)
    return (mix + segment).astype(np.float32), noise_path.name


def scale_to_rms(signal: np.ndarray, target_rms: float) -> np.ndarray:
    current_rms = np.sqrt(np.mean(signal**2) + EPS)
    if current_rms < EPS:
        return signal
    return signal * (target_rms / current_rms)


def peak_normalize(
    signals: Sequence[np.ndarray],
    peak_dbfs: float = -1.0,
) -> Sequence[np.ndarray]:
    peak_target = db_to_amplitude(peak_dbfs)
    peak = max(np.max(np.abs(sig)) for sig in signals)
    if peak < EPS:
        return signals
    if peak <= peak_target:
        return signals
    scale = peak_target / peak
    return [sig * scale for sig in signals]


# ---------------------------------------------------------------------------
# Clip building
# ---------------------------------------------------------------------------


def build_clip(
    assignment: ClipAssignment,
    args: argparse.Namespace,
    asset_manager: AssetManager,
    out_root: Path,
    logger: logging.Logger,
) -> Dict[str, object]:
    spec = assignment.spec
    rng = np.random.default_rng(spec.rng_seed)
    sr = args.target_sr
    total_samples = int(round(CLIP_DURATION_SECONDS * sr))

    silence_floor_dbfs = rng.uniform(*DEFAULT_PAD_DB_RANGE)
    use_external_noise = bool(asset_manager.noise_files) and not args.dry_run
    base_noise1, _ = generate_low_level_noise(
        total_samples, silence_floor_dbfs, rng, asset_manager, use_external=use_external_noise
    )
    base_noise2, _ = generate_low_level_noise(
        total_samples, silence_floor_dbfs, rng, asset_manager, use_external=use_external_noise
    )

    segment_samples = int(round(spec.coverage_ratio * total_samples))
    segment_samples = min(segment_samples, total_samples)
    segment_samples = max(segment_samples, int(0.5 * sr))

    track1 = base_noise1.copy()
    track2 = base_noise2.copy()

    rir_id_1 = "NONE"
    rir_id_2 = "NONE"
    bg_noise_id = "NONE"
    overlap_label = "NA"
    inter_snr = 0.0
    overlap_samples = 0
    s1_start_samples = 0
    s1_end_samples = 0
    s2_start_samples = 0
    s2_end_samples = 0

    if spec.num_speakers == 1:
        speaker1_wav = rng.choice(assignment.speaker1.wavs)
        speech1 = extract_speech_segment(
            speaker1_wav,
            target_sr=sr,
            segment_samples=segment_samples,
            rng=rng,
            logger=logger,
        )
        start1 = placement_to_start(spec.placement, len(speech1), total_samples, rng)
        end1 = min(total_samples, start1 + len(speech1))
        track1[start1:end1] += speech1[: end1 - start1]
        s1_start_samples = start1
        s1_end_samples = end1

        track1 = scale_to_rms(track1, rng.uniform(*TARGET_SOURCE_RMS_RANGE))
        track2 = scale_to_rms(
            track2, db_to_amplitude(silence_floor_dbfs) * 0.5
        )
        if not args.dry_run:
            track1, rir_id_1 = maybe_apply_rir(track1, rng, asset_manager)
            track2, rir_id_2 = maybe_apply_rir(track2, rng, asset_manager)
    else:
        speaker1_wav = rng.choice(assignment.speaker1.wavs)
        speaker2 = assignment.speaker2
        if speaker2 is None:
            raise RuntimeError("Two-speaker spec missing secondary speaker.")
        speaker2_wav = rng.choice(speaker2.wavs)
        speech1 = extract_speech_segment(
            speaker1_wav,
            target_sr=sr,
            segment_samples=segment_samples,
            rng=rng,
            logger=logger,
        )
        speech2 = extract_speech_segment(
            speaker2_wav,
            target_sr=sr,
            segment_samples=segment_samples,
            rng=rng,
            logger=logger,
        )

        if args.balance_overlap:
            target_bucket = spec.overlap_label
            target_ratio = min(sample_overlap_ratio(target_bucket, rng), 0.8)

            min_active_sec = 0.5
            min_active_samples = int(round(min_active_sec * sr))
            if len(speech1) < min_active_samples:
                speech1 = np.pad(speech1, (0, min_active_samples - len(speech1)), mode="edge")
            if len(speech2) < min_active_samples:
                speech2 = np.pad(speech2, (0, min_active_samples - len(speech2)), mode="edge")

            s1_dur_sec = len(speech1) / sr
            s2_dur_sec = len(speech2) / sr
            desired_overlap_sec = min(
                target_ratio * CLIP_DURATION_SECONDS,
                s1_dur_sec,
                s2_dur_sec,
            )
            total_active_sec = s1_dur_sec + s2_dur_sec - desired_overlap_sec
            if total_active_sec <= 0:
                desired_overlap_sec = 0.0
                total_active_sec = s1_dur_sec + s2_dur_sec
            if total_active_sec > CLIP_DURATION_SECONDS:
                scale = CLIP_DURATION_SECONDS / total_active_sec
                s1_dur_sec *= scale
                s2_dur_sec *= scale
                desired_overlap_sec *= scale

            s1_samples_new = int(round(s1_dur_sec * sr))
            s2_samples_new = int(round(s2_dur_sec * sr))
            s1_samples_new = max(min_active_samples, min(s1_samples_new, len(speech1)))
            s2_samples_new = max(min_active_samples, min(s2_samples_new, len(speech2)))
            speech1_used = speech1[:s1_samples_new]
            speech2_used = speech2[:s2_samples_new]
            s1_dur_sec = len(speech1_used) / sr
            s2_dur_sec = len(speech2_used) / sr
            desired_overlap_sec = min(desired_overlap_sec, s1_dur_sec, s2_dur_sec)

            max_start_s1 = max(0.0, CLIP_DURATION_SECONDS - s1_dur_sec)
            max_start_s2 = max(0.0, CLIP_DURATION_SECONDS - s2_dur_sec)

            actual_overlap_sec = 0.0
            actual_overlap_ratio = 0.0
            for attempt in range(3):
                if rng.random() < 0.5:
                    s1_start_sec = float(rng.uniform(0.0, max_start_s1))
                    base_s2_start = s1_start_sec + s1_dur_sec - desired_overlap_sec
                    s2_start_sec = base_s2_start + float(rng.uniform(-0.2, 0.2))
                    s2_start_sec = float(np.clip(s2_start_sec, 0.0, max_start_s2))
                else:
                    s2_start_sec = float(rng.uniform(0.0, max_start_s2))
                    base_s1_start = s2_start_sec + s2_dur_sec - desired_overlap_sec
                    s1_start_sec = base_s1_start + float(rng.uniform(-0.2, 0.2))
                    s1_start_sec = float(np.clip(s1_start_sec, 0.0, max_start_s1))

                s1_end_sec = s1_start_sec + s1_dur_sec
                s2_end_sec = s2_start_sec + s2_dur_sec
                actual_overlap_sec = max(
                    0.0, min(s1_end_sec, s2_end_sec) - max(s1_start_sec, s2_start_sec)
                )
                actual_overlap_ratio = actual_overlap_sec / CLIP_DURATION_SECONDS
                if target_ratio <= 0.5 and actual_overlap_ratio > 0.8 and attempt < 2:
                    continue
                break

            s1_start_samples = int(round(s1_start_sec * sr))
            s2_start_samples = int(round(s2_start_sec * sr))
            s1_start_samples = max(0, min(s1_start_samples, total_samples - len(speech1_used)))
            s2_start_samples = max(0, min(s2_start_samples, total_samples - len(speech2_used)))

            track1[s1_start_samples : s1_start_samples + len(speech1_used)] += speech1_used
            track2[s2_start_samples : s2_start_samples + len(speech2_used)] += speech2_used
            s1_end_samples = s1_start_samples + len(speech1_used)
            s2_end_samples = s2_start_samples + len(speech2_used)
            overlap_samples = max(
                0,
                min(s1_end_samples, s2_end_samples) - max(s1_start_samples, s2_start_samples),
            )
        else:
            start1 = placement_to_start(spec.placement, len(speech1), total_samples, rng)
            end1 = min(total_samples, start1 + len(speech1))
            track1[start1:end1] += speech1[: end1 - start1]
            s1_start_samples = start1
            s1_end_samples = end1

            min_overlap_fraction = feasible_overlap_fraction(len(speech2), total_samples)
            target_min, target_max = spec.overlap_range
            if target_max < min_overlap_fraction:
                overlap_fraction = min_overlap_fraction
            else:
                overlap_fraction = rng.uniform(max(target_min, min_overlap_fraction), target_max)

            desired_overlap = overlap_fraction * len(speech2)
            offset_distance = max(0.0, len(speech2) - desired_overlap)
            sign = 1 if rng.random() < 0.5 else -1
            start2 = int(round(start1 + sign * offset_distance))
            start2 = max(0, min(start2, total_samples - len(speech2)))
            end2 = min(total_samples, start2 + len(speech2))
            track2[start2:end2] += speech2[: end2 - start2]
            s2_start_samples = start2
            s2_end_samples = end2

        overlap_samples = max(
            0,
            min(s1_end_samples, s2_end_samples) - max(s1_start_samples, s2_start_samples),
        )

        inter_snr = rng.uniform(-5.0, 10.0)
        target_rms_s1 = rng.uniform(*TARGET_SOURCE_RMS_RANGE)
        track1 = scale_to_rms(track1, target_rms_s1)
        desired_ratio = db_to_amplitude(inter_snr)
        track2 = scale_to_rms(track2, max(target_rms_s1 / (desired_ratio + EPS), TARGET_SOURCE_RMS_RANGE[0]))
        if not args.dry_run:
            track1, rir_id_1 = maybe_apply_rir(track1, rng, asset_manager)
            track2, rir_id_2 = maybe_apply_rir(track2, rng, asset_manager)

    track1 = track1[:total_samples]
    track2 = track2[:total_samples]

    if spec.num_speakers == 1:
        rir_annotation = f"{rir_id_1}|{rir_id_2}" if rir_id_2 != "NONE" else rir_id_1
    else:
        rir_annotation = f"{rir_id_1}|{rir_id_2}"

    if args.dry_run:
        mix = (track1 + track2).astype(np.float32)
    else:
        mix = (track1 + track2).astype(np.float32)
        mix, bg_noise_id = maybe_apply_background_noise(mix, rng, asset_manager)
        track1, track2, mix = peak_normalize([track1, track2, mix], peak_dbfs=-1.0)

    clip_name = f"mix_{spec.clip_id}.wav"
    split = spec.split
    mix_path = out_root / "mix" / split / clip_name
    s1_path = out_root / "s1" / split / f"s1_{spec.clip_id}.wav"
    s2_path = out_root / "s2" / split / f"s2_{spec.clip_id}.wav"

    if not args.dry_run:
        ensure_dir(mix_path.parent)
        ensure_dir(s1_path.parent)
        ensure_dir(s2_path.parent)
        sf.write(mix_path, mix, sr, subtype="PCM_16")
        sf.write(s1_path, track1.astype(np.float32), sr, subtype="PCM_16")
        sf.write(s2_path, track2.astype(np.float32), sr, subtype="PCM_16")

    s1_start_sec = round(s1_start_samples / sr, 3)
    s1_dur_sec = round(max(0, s1_end_samples - s1_start_samples) / sr, 3)
    if spec.num_speakers == 2:
        s2_start_sec_raw = s2_start_samples / sr
        s2_dur_sec_raw = max(0, s2_end_samples - s2_start_samples) / sr
        s2_start_sec = round(s2_start_sec_raw, 3)
        s2_dur_sec = round(s2_dur_sec_raw, 3)
        overlap_sec_value = overlap_samples / sr
        overlap_sec = round(overlap_sec_value, 3)
        overlap_ratio_window_value = overlap_sec_value / CLIP_DURATION_SECONDS
        overlap_ratio_window = round(overlap_ratio_window_value, 4)
        overlap_label = categorize_overlap(overlap_ratio_window_value)
    else:
        s2_start_sec = 0.0
        s2_dur_sec = 0.0
        overlap_sec = 0.0
        overlap_ratio_window = 0.0
        overlap_label = "NA"

    row = {
        "id": spec.clip_id,
        "mix_wav": relative_to(mix_path, out_root),
        "s1_wav": relative_to(s1_path, out_root),
        "s2_wav": relative_to(s2_path, out_root),
        "num_speakers": spec.num_speakers,
        "overlap_bucket": overlap_label if spec.num_speakers == 2 else "NA",
        "coverage_ratio": spec.coverage_ratio,
        "placement": spec.placement,
        "s1_start": s1_start_sec,
        "s1_dur": s1_dur_sec,
        "s2_start": s2_start_sec,
        "s2_dur": s2_dur_sec,
        "overlap_sec": overlap_sec,
        "overlap_ratio_window": overlap_ratio_window,
        "snr_db": round(inter_snr, 3) if spec.num_speakers == 2 else 0.0,
        "rir_id": rir_annotation,
        "bg_noise_id": bg_noise_id,
        "gender_s1": assignment.speaker1.gender if assignment.speaker1 else "NONE",
        "gender_s2": assignment.speaker2.gender if assignment.speaker2 else "NONE",
        "spk_id_s1": assignment.speaker1.speaker_id if assignment.speaker1 else "NONE",
        "spk_id_s2": assignment.speaker2.speaker_id if assignment.speaker2 else "NONE",
        "institute_s1": assignment.speaker1.institute if assignment.speaker1 else "NONE",
        "institute_s2": assignment.speaker2.institute if assignment.speaker2 else "NONE",
        "silence_floor_dbfs": round(silence_floor_dbfs, 3),
        "sample_rate": sr,
    }
    return row


# ---------------------------------------------------------------------------
# Metadata, validation, orchestration
# ---------------------------------------------------------------------------


def write_metadata_csv(out_root: Path, split: str, rows: List[Dict[str, object]]) -> None:
    metadata_dir = out_root / "metadata"
    ensure_dir(metadata_dir)
    csv_path = metadata_dir / f"{split}.csv"
    if not rows:
        columns = [
            "id",
            "mix_wav",
            "s1_wav",
            "s2_wav",
            "num_speakers",
            "overlap_bucket",
            "coverage_ratio",
            "placement",
            "s1_start",
            "s1_dur",
            "s2_start",
            "s2_dur",
            "overlap_sec",
            "overlap_ratio_window",
            "snr_db",
            "rir_id",
            "bg_noise_id",
            "gender_s1",
            "gender_s2",
            "spk_id_s1",
            "spk_id_s2",
            "institute_s1",
            "institute_s2",
            "silence_floor_dbfs",
            "sample_rate",
        ]
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)


def validate_dataset(
    out_root: Path,
    metadata: Dict[str, List[Dict[str, object]]],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    if not args.dry_run:
        sr = args.target_sr
        expected_duration = CLIP_DURATION_SECONDS
        tolerance = 0.01
        for split, rows in metadata.items():
            for row in rows:
                mix_path = out_root / row["mix_wav"]
                if not mix_path.exists():
                    raise FileNotFoundError(f"Missing mix file: {mix_path}")
                info = sf.info(str(mix_path))
                if info.samplerate != sr:
                    raise RuntimeError(f"Unexpected sample rate in {mix_path}: {info.samplerate}")
                if abs(info.duration - expected_duration) > tolerance:
                    raise RuntimeError(f"Unexpected duration in {mix_path}: {info.duration}")
    else:
        logger.info("DRY RUN enabled: skipped audio validations.")

    split_speakers_sets: Dict[str, set] = {}
    for split, rows in metadata.items():
        spks = set()
        for row in rows:
            spks.add(row["spk_id_s1"])
            if row["spk_id_s2"] != "NONE":
                spks.add(row["spk_id_s2"])
        split_speakers_sets[split] = spks
    splits = list(split_speakers_sets.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_speakers_sets[splits[i]] & split_speakers_sets[splits[j]]
            if overlap:
                raise RuntimeError(
                    f"Speaker leakage between {splits[i]} and {splits[j]}: {sorted(overlap)[:5]}"
                )
    logger.info("Validation complete: file checks and speaker leakage OK.")


def gather_asset_files(path: Optional[str]) -> List[Path]:
    if not path:
        return []
    base = Path(path)
    if not base.exists():
        return []
    audio_exts = {".wav", ".flac", ".ogg"}
    files = [p for p in base.rglob("*") if p.suffix.lower() in audio_exts]
    return files


def build_dataset(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    ensure_dir(out_root)
    logger = setup_logging(out_root)
    set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.dry_run:
        logger.info("DRY RUN enabled: audio files will not be written.")

    logger.info("Starting dataset build with total_clips=%d", args.total_clips)
    speakers = discover_speakers(Path(args.tcc_root), logger)
    split_map = split_speakers(speakers, rng, logger)
    overlap_weights = BALANCED_OVERLAP_WEIGHTS if args.balance_overlap else DEFAULT_OVERLAP_WEIGHTS
    clip_specs = plan_clip_specs(args.total_clips, rng, logger, overlap_weights)
    assignments = assign_speakers_to_specs(clip_specs, split_map, rng, logger)

    noise_files = gather_asset_files(args.noise_dir)
    rir_files = gather_asset_files(args.rir_dir)
    if noise_files:
        logger.info("Found %d noise assets.", len(noise_files))
    else:
        logger.info("No noise assets found; using procedural noise for fillers.")
    if rir_files:
        logger.info("Found %d RIR assets.", len(rir_files))
    else:
        logger.info("No RIR assets provided; skipping reverberation.")

    asset_manager = AssetManager(
        target_sr=args.target_sr,
        noise_files=noise_files,
        rir_files=rir_files,
    )

    metadata_rows: Dict[str, List[Dict[str, object]]] = {split: [] for split, _ in SPLIT_CONFIG}
    requested_overlap_counter: Dict[str, Counter] = defaultdict(Counter)
    actual_overlap_counter: Dict[str, Counter] = defaultdict(Counter)

    for split, assignments_list in assignments.items():
        if not assignments_list:
            continue
        progress = tqdm(total=len(assignments_list), desc=f"{split} clips", unit="clip")
        rows: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(build_clip, assignment, args, asset_manager, out_root, logger): assignment
                for assignment in assignments_list
            }
            for future in as_completed(future_map):
                assignment = future_map[future]
                try:
                    row = future.result()
                    rows.append(row)
                    if assignment.spec.num_speakers == 2:
                        requested_overlap_counter[split][assignment.spec.overlap_label] += 1
                        actual_overlap_counter[split][row["overlap_bucket"]] += 1
                    progress.update(1)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("Failed to build clip %s: %s", assignment.spec.clip_id, exc)
                    raise
        progress.close()
        rows.sort(key=lambda r: int(r["id"]))
        metadata_rows[split] = rows
        write_metadata_csv(out_root, split, rows)
        logger.info("Completed split %s. Rows=%d", split, len(rows))

    logger.info("Requested vs actual overlap summary:")
    for split in metadata_rows.keys():
        req = requested_overlap_counter.get(split, Counter())
        actual = actual_overlap_counter.get(split, Counter())
        logger.info("Split %s requested: %s", split, dict(req))
        logger.info("Split %s actual   : %s", split, dict(actual))

    all_rows: List[Dict[str, object]] = [row for rows in metadata_rows.values() for row in rows]
    if all_rows:
        overlap_values = [float(row["overlap_sec"]) for row in all_rows]
        min_overlap = min(overlap_values)
        max_overlap = max(overlap_values)
        mean_overlap = float(np.mean(overlap_values))
        logger.info(
            "Overlap stats (sec): min=%.3f, mean=%.3f, max=%.3f",
            min_overlap,
            mean_overlap,
            max_overlap,
        )
        all_rows_with_split: List[Tuple[str, Dict[str, object]]] = []
        for split, rows in metadata_rows.items():
            for row in rows:
                all_rows_with_split.append((split, row))
        two_rows = [item for item in all_rows_with_split if item[1]["num_speakers"] == 2]
        sample_source = two_rows if len(two_rows) >= 10 else all_rows_with_split
        rng_debug = np.random.default_rng(args.seed + 101)
        sample_count = min(10, len(sample_source))
        if sample_count > 0:
            indices = rng_debug.choice(len(sample_source), size=sample_count, replace=False)
            for idx in indices:
                split, row = sample_source[int(idx)]
                logger.info(
                    "Overlap sample split=%s id=%s s1_start=%.3f s1_dur=%.3f s2_start=%.3f s2_dur=%.3f overlap_sec=%.3f ratio=%.3f",
                    split,
                    row["id"],
                    float(row["s1_start"]),
                    float(row["s1_dur"]),
                    float(row["s2_start"]),
                    float(row["s2_dur"]),
                    float(row["overlap_sec"]),
                    float(row["overlap_ratio_window"]),
                )

    validate_dataset(out_root, metadata_rows, args, logger)
    logger.info("Dataset build finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build two-speaker mixtures from TCC-300.")
    parser.add_argument("--tcc_root", type=str, required=True, help="Path to TCC-300 root directory.")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory.")
    parser.add_argument("--total_clips", type=int, default=30000, help="Total number of clips to generate.")
    parser.add_argument("--seed", type=int, default=137, help="Random seed.")
    parser.add_argument("--rir_dir", type=str, default=None, help="Directory with impulse responses (optional).")
    parser.add_argument("--noise_dir", type=str, default=None, help="Directory with background/noise assets (optional).")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2), help="Parallel workers.")
    parser.add_argument(
        "--balance_overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable balanced overlap scheduling (use --no-balance-overlap to disable).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write metadata/logs only; skip audio DSP and WAV exports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
