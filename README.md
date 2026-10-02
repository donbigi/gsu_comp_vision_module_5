# Module 5 — Temporal Closed-Loop Verification & Recovery (feasibility prototype)

A minimal prototype for validating the "temporal closed-loop verification and
recovery" idea in a Hierarchical VLA framework: after a low-level policy
executes a sub-goal, a **Qwen2.5-VL** video-language model watches the workspace
clip and decides `SUCCESS` / `FAILURE`, producing a corrective sub-goal on
failure.

## Run it yourself

```bash
git clone https://github.com/<your-org>/<repo>.git
cd <repo>
pip install torch torchvision transformers accelerate av qwen_vl_utils
python - <<'PY'
from verifier import VideoSubGoalVerifier
v = VideoSubGoalVerifier()
print(v.verify_action_execution("https://example.com/clip.mp4", "Grasp the cup"))
PY
```

## Files

- `verifier.py` — `VideoSubGoalVerifier` (the core Video-LLM check) plus a
  `ClosedLoopVLAController` skeleton that inserts recovery steps on failure.

## Setup

```bash
pip install torch torchvision transformers accelerate av qwen_vl_utils
```

The verifier defaults to `Qwen/Qwen2.5-VL-3B-Instruct` on **CPU**. On this M4
Mac (16 GB unified memory) the 7B checkpoint does not fit, and bf16-on-MPS
produces garbled tokens, so CPU is the safe default. Override the device with
the `QWEN_DEVICE` env var (`mps` only with `torch_dtype=float16`).

## Getting a sample video (do **not** commit video to git)

`VideoSubGoalVerifier.verify_action_execution(video_path=...)` accepts **either**
a local file path **or a direct `https://…` URL** — `qwen_vl_utils` streams
http(s) URLs straight through torchvision's PyAV backend, so you can validate
the verifier without downloading or committing anything:

```python
verifier.verify_action_execution(
    video_path="https://example.com/path/to/robot_clip.mp4",
    attempted_subgoal="Grasp the cup from the coffee machine",
)
```

To work from a **local** clip instead (still gitignored), pull one real robot
manipulation clip from a public source, e.g. the
[LeRobot](https://github.com/huggingface/lerobot) datasets on Hugging Face, or
[DROID / Open X-Embodiment](https://robotics-transformer-x.github.io/):

```bash
# example: fetch a LeRobot episode clip with the HF CLI / datasets library
huggingface-cli download lerobot/<dataset> --include "*.mp4" --local-dir ./sample_videos
```

The clip must be decodable by torchvision's PyAV backend. If you re-encode an
AV1 / high-fps source, target H.264 at 30 fps:

```bash
ffmpeg -y -ss <start_s> -t <dur_s> -i source.mp4 \
  -c:v libx264 -preset fast -pix_fmt yuv420p -vf "fps=30" -an clip.mp4
```

All `*.mp4` / image files are ignored via `.gitignore`, so any downloaded clip
stays out of version control.

## Running

```python
from verifier import VideoSubGoalVerifier

verifier = VideoSubGoalVerifier()
result = verifier.verify_action_execution(
    video_path="sample_videos/clip.mp4",
    attempted_subgoal="Grasp the cup from the coffee machine",
)
print(result)
```

`verify_action_execution` returns JSON:

```json
{
  "status": "SUCCESS" | "FAILURE",
  "detected_outcome": "...",
  "failure_reason": "NONE" | "slipped_grasp" | "missed_target" | "dropped_object" | "...",
  "corrective_subgoal": "NONE" | "<atomic recovery step>"
}
```

`verifier.py`'s `controller.execute_plan(sandwich_plan)` is left commented out; wire
it to your real low-level VLA dispatch and workspace-camera recorder
(`rollout_clips/step_<i>.mp4`) to exercise the full closed loop.
# gsu_comp_vision_module_5
