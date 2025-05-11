# pipelines/orchestrator.py
import json, uuid, pathlib, datetime as dt
import torchaudio
from modules.separation.separator import AudioSeparator
from modules.speaker_id.v4 import SpeakerIdentifier
from modules.asr.whisper_asr import WhisperASR

sep = AudioSeparator()
spk = SpeakerIdentifier()
asr = WhisperASR(model_name="medium", gpu=False)

def run_pipeline(raw_wav: str):
    # 1️⃣ 先 load WAV → 得到 tensor
    waveform, sr = torchaudio.load(raw_wav)

    # 2️⃣ 設定好輸出資料夾（可以按日期命名），或傳進原本的 separate_and_save
    out_dir = pathlib.Path("work_output") / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3️⃣ 呼叫核心分離函式，傳入 audio_tensor、output_dir、segment_index
    #    這裡我們單檔只用 segment_index=0，你也可以做迴圈多檔
    segments = sep.separate_and_save(waveform, str(out_dir), segment_index=0)
    # segments = [(seg_path, start, end), …]

    # 4️⃣ 逐一做語者辨識 & ASR，並加進 bundle
    bundle = []
    
    for seg_path, t0, t1 in segments:
        # 正確解包：第一個是 speaker_id、第二個才是 name、第三個才是 distance
        result = spk.process_audio_file(seg_path)
        if result is None:
            # 如果意外回傳 None，可以選擇跳過或預設值
            continue

        speaker_id, name, dist = result
        dist = float(dist)  # 這回 dist 就是數值，不會再炸

        # 接著再做 ASR
        text, conf = asr.transcribe(seg_path)

        bundle.append({
            "start": round(t0, 2),
            "end":   round(t1, 2),
            "speaker": name,
            "distance": round(dist, 3),
            "text": text,
            "confidence": round(conf, 2)
        })



    # 5️⃣ 寫檔
    json_path = out_dir / "output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    print("✅ Pipeline finished →", json_path)
    return bundle

if __name__ == "__main__":
    import sys
    run_pipeline(sys.argv[1])

