from modules.asr.whisper_asr import WhisperASR
from utils.constants import DEFAULT_WHISPER_MODEL, DEFAULT_WHISPER_BEAM_SIZE

asr = WhisperASR(model_name=DEFAULT_WHISPER_MODEL, gpu=True, beam=DEFAULT_WHISPER_BEAM_SIZE, lang="auto")
asr.transcribe_dir("SepOutput", "conv001")
