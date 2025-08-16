# stats_summarizer.py
import pandas as pd
import numpy as np

AUDIO_LEN_SEC = 6.0       # 固定音檔長度（秒）
LOW_CONF_THRESHOLD = 0.6  # Low-confidence 門檻

def compute_summary(df: pd.DataFrame) -> dict:
    out = {}

    # --- CER：把 cer1 + cer2 併在一起統計 ---
    cer_all = np.concatenate([df["cer1"].values, df["cer2"].values])
    out["cer_mean"] = float(np.mean(cer_all))
    out["cer_median"] = float(np.median(cer_all))
    out["cer_p90"] = float(np.percentile(cer_all, 90))
    out["cer_std"] = float(np.std(cer_all, ddof=0))

    # --- SI-SDR：每列只取「正的那個」；兩個都正就取最大；兩個都<=0 就跳過 ---
    sisdr_pick = []
    for s1, s2 in zip(df["si_sdr_src1"], df["si_sdr_src2"]):
        candidates = [x for x in (s1, s2) if x > 0]
        if len(candidates) == 1:
            sisdr_pick.append(candidates[0])
        elif len(candidates) > 1:
            sisdr_pick.append(max(candidates))
        # 若兩個都 <= 0，則忽略（不列入平均）
    out["sisdr_mean_pos_only"] = float(np.mean(sisdr_pick)) if sisdr_pick else float("nan")
    out["sisdr_count_used"] = int(len(sisdr_pick))
    out["sisdr_rows_total"] = int(len(df))

    # --- ΔSI-SDR：直接把兩欄併在一起平均（你的規則） ---
    delta_all = np.concatenate([df["delta_si_sdr_src1"].values, df["delta_si_sdr_src2"].values])
    out["delta_sisdr_mean"] = float(np.mean(delta_all))
    out["delta_sisdr_median"] = float(np.median(delta_all))
    out["delta_sisdr_p90"] = float(np.percentile(delta_all, 90))
    out["delta_sisdr_std"] = float(np.std(delta_all, ddof=0))

    # --- Confidence ---
    out["avg_conf_mean"] = float(df["avg_conf"].mean())
    out["low_conf_pct@0.6"] = float((df["avg_conf"] < LOW_CONF_THRESHOLD).mean() * 100.0)

    # --- Accuracy 分桶 + 平均 ---
    acc = df["accuracy"]
    out["accuracy_mean"] = float(acc.mean())
    out["acc_1_pct"] = float((acc == 1).mean() * 100.0)
    out["acc_0_5_pct"] = float((acc == 0.5).mean() * 100.0)
    out["acc_0_pct"] = float((acc == 0).mean() * 100.0)

    # --- RTF 與耗時占比 ---
    rtf = df["total_time"] / AUDIO_LEN_SEC
    out["rtf_mean"] = float(rtf.mean())
    out["rtf_median"] = float(rtf.median())
    out["rtf_p90"] = float(np.percentile(rtf, 90))
    out["rtf_std"] = float(rtf.std(ddof=0))

    total = df["total_time"].replace({0: np.nan})
    out["share_sep_pct_mean"] = float(((df["sep_time"] / total) * 100.0).mean())
    out["share_sid_pct_mean"] = float(((df["sid_time"] / total) * 100.0).mean())
    out["share_asr_pct_mean"] = float(((df["asr_time"] / total) * 100.0).mean())

    return out

def summarize_csv(in_csv: str, out_csv: str) -> None:
    df = pd.read_csv(in_csv)
    s = compute_summary(df)
    summary_df = pd.DataFrame({
        "Metric": [
            "CER mean", "CER median", "CER p90", "CER std",
            "SISDR (pos-only) mean", "SISDR used rows / total",
            "ΔSI-SDR mean", "ΔSI-SDR median", "ΔSI-SDR p90", "ΔSI-SDR std",
            "Avg confidence", "Low-confidence % @0.6",
            "Accuracy mean", "Acc=1 %", "Acc=0.5 %", "Acc=0 %",
            "RTF mean", "RTF median", "RTF p90", "RTF std",
            "Share: Separation % (mean)",
            "Share: SpeakerID % (mean)",
            "Share: ASR % (mean)",
        ],
        "Value": [
            s["cer_mean"], s["cer_median"], s["cer_p90"], s["cer_std"],
            s["sisdr_mean_pos_only"], f'{s["sisdr_count_used"]} / {s["sisdr_rows_total"]}',
            s["delta_sisdr_mean"], s["delta_sisdr_median"], s["delta_sisdr_p90"], s["delta_sisdr_std"],
            s["avg_conf_mean"], s["low_conf_pct@0.6"],
            s["accuracy_mean"], s["acc_1_pct"], s["acc_0_5_pct"], s["acc_0_pct"],
            s["rtf_mean"], s["rtf_median"], s["rtf_p90"], s["rtf_std"],
            s["share_sep_pct_mean"], s["share_sid_pct_mean"], s["share_asr_pct_mean"],
        ]
    })
    summary_df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", required=True, help="Path to pipeline_results.csv")
    p.add_argument("--out_csv", default="pipeline_summary.csv", help="Where to write the summary CSV")
    args = p.parse_args()
    summarize_csv(args.in_csv, args.out_csv)
    print(f"Saved summary to: {args.out_csv}")
