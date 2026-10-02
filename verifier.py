import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class VideoSubGoalVerifier:
    """
    Evaluates short execution video clips from the robot's workspace camera
    to verify sub-goal completion and route corrective actions.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        # NOTE: default switched from 7B -> 3B. The 7B checkpoint (~15GB in bf16)
        # exceeds the 16GB unified memory on this M4 Mac; 3B (~6GB) fits comfortably.
        print(f"Loading {model_id} for temporal verification...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        # NOTE: device_map="auto" mis-dispatches on Apple Silicon (it silently
        # falls back to disk offload, which is unusably slow), so pick the device
        # explicitly. We default to CPU because bf16 on MPS produced corrupted/
        # garbled tokens for this model on this machine; MPS is faster but only
        # use it with QWEN_DEVICE=mps and (ideally) torch_dtype=float16.
        device = os.environ.get("QWEN_DEVICE") or "cpu"
        self.model = self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def verify_action_execution(self, video_path: str, attempted_subgoal: str) -> dict:
        prompt = f"""
You are the verification module of an autonomous robot.
The robot attempted to complete the following sub-goal:
"{attempted_subgoal}"

Analyze the provided execution video and evaluate whether the action succeeded or failed.
Provide your response strictly in valid JSON format:
{{
    "status": "SUCCESS" | "FAILURE",
    "detected_outcome": "<short description of physical state observed>",
    "failure_reason": "<NONE or specific physical failure such as slipped_grasp, missed_target, dropped_object>",
    "corrective_subgoal": "<NONE if SUCCESS, or the atomic sub-goal needed to recover>"
}}
"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 2.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        # Parse JSON output from the model
        try:
            cleaned_json = response_text[response_text.find("{"):response_text.rfind("}") + 1]
            return json.loads(cleaned_json)
        except Exception:
            return {"status": "UNKNOWN", "raw_output": response_text}


class ClosedLoopVLAController:
    """
    Coordinates high-level task plans with low-level execution and Video-LLM verification.
    """
    def __init__(self, verifier: VideoSubGoalVerifier):
        self.verifier = verifier

    def execute_plan(self, plan: list[str]):
        step_idx = 0
        max_retries = 2
        retry_counts = {step: 0 for step in plan}

        while step_idx < len(plan):
            current_subgoal = plan[step_idx]
            print(f"\n[Executing Step {step_idx + 1}/{len(plan)}]: '{current_subgoal}'")

            # In practice, dispatch to low-level VLA (e.g., pi0 / PaliGemma action chunker)
            # and record execution clip:
            mock_video_path = f"rollout_clips/step_{step_idx}.mp4"

            # Temporal verification using the Video-LLM
            verification = self.verifier.verify_action_execution(
                video_path=mock_video_path,
                attempted_subgoal=current_subgoal
            )

            print(f" Verification Status: {verification.get('status')}")
            print(f" Observed State:     {verification.get('detected_outcome')}")

            if verification.get("status") == "SUCCESS":
                step_idx += 1
            else:
                retry_counts[current_subgoal] += 1
                if retry_counts[current_subgoal] > max_retries:
                    print(f"[Alert] Max retries reached for '{current_subgoal}'. Requesting human assistance.")
                    break

                recovery_step = verification.get("corrective_subgoal", current_subgoal)
                print(f"[Recovery Initiated]: Inserting corrective step: '{recovery_step}'")
                plan.insert(step_idx, recovery_step)


# Example invocation
if __name__ == "__main__":
    verifier = VideoSubGoalVerifier()
    controller = ClosedLoopVLAController(verifier)

    sandwich_plan = [
        "Pick up bottom bread slice and place on plate",
        "Grasp tomato slice and place on bread",
        "Pick up top bread slice and close sandwich"
    ]

    # controller.execute_plan(sandwich_plan)
