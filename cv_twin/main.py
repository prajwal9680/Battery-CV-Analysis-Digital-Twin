from cv_twin.qc.preprocess import PreprocessConfig, load_and_preprocess

cfg = PreprocessConfig(
    smooth_window_pts=31,
    smooth_polyorder=3,
    spike_zscore_thresh=4.5,
    points_per_volt=800,
    baseline_tail_frac=0.12,
    voltage_scale=1.0,
    current_scale=1.0
)

prep = load_and_preprocess("cv_twin/data/sample/sample_cv.csv", cfg)
df_clean = prep["df_clean"]
segments_meta = prep["segments"]

print(df_clean.head())
