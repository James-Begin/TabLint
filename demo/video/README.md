# Demo-video data

- `auto_mpg.csv`: the Auto MPG data set by R. Quinlan, UCI Machine Learning Repository,
  https://doi.org/10.24432/C5859H, licensed CC BY 4.0. Converted from `auto-mpg.data`: the origin code is mapped to
  USA/Europe/Japan and the year to four digits; values are otherwise unchanged.
- `auto_mpg_dirty.csv`: the same table with **8 errors we planted on purpose**, listed with their stories in
  `answer_key.json`.
- `*.proofread.json`: stored Proofread (TabPFN-3.5) reports. Open them with `proofread view`, in the app, the notebook or
  VS Code, with no GPU.

Regenerate with `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python benchmarks/make_video_demos.py`.
