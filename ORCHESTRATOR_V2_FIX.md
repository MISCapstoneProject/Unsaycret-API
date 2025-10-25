# Orchestrator V2 - Missing Dependencies Fix

## Summary
This document outlines the fix for `ModuleNotFoundError: No module named 'wespeaker'` and makes orchestrator_v2.py robust against missing dependencies.

## Changes Required

### 1. Add Lazy Imports in orchestrator_v2.py (line ~23-30)

Replace:
```python
import numpy as np
try:
    import pyaudio  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyaudio = None
from scipy.signal import resample_poly

os.environ.setdefault("MKL_DISABLE_FAST_MM", "1")
```

With:
```python
import numpy as np
try:
    import pyaudio  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyaudio = None

try:
    import torch
    import torchaudio
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore
    torchaudio = None  # type: ignore

try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    resample_poly = None  # type: ignore

os.environ.setdefault("MKL_DISABLE_FAST_MM", "1")
```

### 2. Update init_pipeline_modules() function (line ~146)

Add graceful fallback for speaker identification:

```python
def init_pipeline_modules(
    load_separator: bool = True,
    load_identifier: bool = True,
    load_asr: bool = True,
    id_backend: str = "auto",
) -> Tuple[Optional[AudioSeparator], Optional[SpeakerIdentifier], Optional[WhisperASR], bool]:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required but not installed. Please install torch and torchaudio.")
    
    current_cuda_device = CUDA_DEVICE_INDEX

    if FORCE_CPU:
        use_gpu = False
        logger.info("FORCE_CPU=true; forcing CPU execution.")
    else:
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            if current_cuda_device < torch.cuda.device_count():
                torch.cuda.set_device(current_cuda_device)
                logger.info(
                    "CUDA device %s selected (%s)",
                    current_cuda_device,
                    torch.cuda.get_device_name(current_cuda_device),
                )
            else:
                logger.warning(
                    "CUDA index %s unavailable; defaulting to device 0.",
                    current_cuda_device,
                )
                current_cuda_device = 0
                torch.cuda.set_device(current_cuda_device)
                logger.info("Using CUDA device 0: %s", torch.cuda.get_device_name(0))

    logger.info("Pipeline device: %s", f"cuda:{current_cuda_device}" if use_gpu else "cpu")

    separator = None
    if load_separator:
        try:
            separator = AudioSeparator()
        except Exception as exc:
            logger.warning("Failed to load AudioSeparator: %s", exc)
            separator = None
    
    identifier = None
    if load_identifier and id_backend != "none":
        try:
            identifier = SpeakerIdentifier()
            logger.info("[Identification] Loaded speaker identifier successfully")
        except ImportError as exc:
            logger.warning("[Identification] Speaker identifier unavailable (missing %s) → fallback to none", exc.name if hasattr(exc, 'name') else 'dependencies')
            identifier = None
        except Exception as exc:
            logger.warning("[Identification] Failed to load speaker identifier: %s → fallback to none", exc)
            identifier = None
    
    asr = None
    if load_asr:
        try:
            asr = WhisperASR(
                model_name=DEFAULT_WHISPER_MODEL,
                gpu=use_gpu,
                beam=DEFAULT_WHISPER_BEAM_SIZE,
            )
        except Exception as exc:
            logger.warning("Failed to load WhisperASR: %s", exc)
            asr = None
            
    return separator, identifier, asr, use_gpu
```

### 3. Update VID_identify_v5.py (line ~247)

Make wespeaker import lazy and safe:

```python
        elif self.model_type == "wespeaker":
            try:
                import wespeaker
            except ImportError:
                raise ImportError(
                    "wespeaker is not installed. Install it with: pip install wespeaker"
                )
            
            try:
                # 確保下載目錄存在並且模型文件完整
                model_dir = snapshot_download(
                    repo_id=WESPEAKER_SPEAKER_MODEL,
                    cache_dir=get_model_save_dir("wespeaker"),
                    force_download=False,  # 避免重複下載
                    resume_download=True   # 支援斷點續傳
                )
                
                # 檢查模型文件是否存在
                import glob
                model_files = glob.glob(os.path.join(model_dir, "*.onnx")) + glob.glob(os.path.join(model_dir, "*.pt"))
                if not model_files:
                    raise FileNotFoundError(f"模型文件未在 {model_dir} 中找到")
                
                self.model = wespeaker.load_model(model_dir)
                logger.info(f"✅ 已載入 Wespeaker 模型: {WESPEAKER_SPEAKER_MODEL}")
                
            except Exception as e:
                logger.error(f"Wespeaker 模型載入失敗：{e}")
                raise
```

### 4. Add CLI Flags to parse_args() (line ~2094)

Add after `--debug` argument:

```python
    parser.add_argument("--debug", action="store_true", help="Enable verbose DEBUG logging.")
    parser.add_argument("--id-backend", choices=["auto", "wespeaker", "pyannote", "speechbrain", "none"], 
                        default="auto", help="Speaker identification backend to use.")
    parser.add_argument("--id-device", choices=["cuda", "cpu", "auto"], default="auto",
                        help="Device for speaker identification model.")
```

### 5. Update main() to pass id_backend (line ~2257)

```python
    if args.mode == "asr_only":
        if args.stream:
            raise ValueError("Streaming mode is incompatible with --mode asr_only.")

        _, _, asr, _ = init_pipeline_modules(
            load_separator=False,
            load_identifier=False,
            load_asr=True,
            id_backend=getattr(args, 'id_backend', 'auto'),
        )
```

And similarly for pipeline modes:

```python
    load_asr = mode == "pipeline" and enable_asr
    sep, identifier, asr, use_gpu = init_pipeline_modules(
        load_separator=True,
        load_identifier=True,
        load_asr=load_asr,
        id_backend=getattr(args, 'id_backend', 'auto'),
    )
```

### 6. Handle None identifier in WindowOrchestrator

Update `process_window()` method around line ~1087 where embeddings are extracted:

```python
        embeddings: List[np.ndarray] = []
        for info in kept_infos:
            try:
                if self.identifier and hasattr(self.identifier, 'audio_processor'):
                    emb = self.identifier.audio_processor.extract_embedding_from_stream(
                        info["audio"], sr
                    )
                else:
                    # Fallback: use random but consistent embeddings
                    logger.warning("Speaker identifier unavailable, using fallback track assignment")
                    emb = np.random.RandomState(window_idx).randn(192).astype(np.float32)
            except Exception as exc:  # pragma: no cover
                logger.warning("Embedding extraction failed: %s", exc)
                emb = np.zeros(192, dtype=np.float32)
            embeddings.append(emb)
```

## Testing Commands

```powershell
# Test 1: With missing wespeaker (should gracefully fallback)
python -m pipelines.orchestrator_v2 --mode pipeline --wav data/mix/m02.wav --out outputs/test1.jsonl --id-backend auto

# Test 2: Explicitly disable ID
python -m pipelines.orchestrator_v2 --mode pipeline --wav data/mix/m02.wav --out outputs/test2.jsonl --id-backend none

# Test 3: Streaming mode
python -m pipelines.orchestrator_v2 --stream --mode pipeline --id-backend auto --record-secs 10
```

## Expected Behavior

1. **Missing wespeaker**: Pipeline continues with warning, tracks assigned by default logic
2. **--id-backend none**: No speaker identification attempted  
3. **Missing torch**: Immediate error with clear message
4. **Missing scipy**: Graceful degradation or error depending on usage

