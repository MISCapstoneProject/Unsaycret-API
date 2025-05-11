from modules.asr.whisper_asr import WhisperASR

asr = WhisperASR(model_name="medium", gpu=True, beam=5, lang="auto")
asr.transcribe_dir("SepOutput", "conv001")
