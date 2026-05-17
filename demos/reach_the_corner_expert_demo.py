from generalization_grid_games.envs import reach_the_corner as rtc
from generalization_grid_games.envs.utils import run_plan_demo
from pathlib import Path


if __name__ == "__main__":
    outdir = Path(__file__).resolve().parent / "out"
    video_path = run_plan_demo(
        rtc.ReachTheCornerGymEnv3,
        rtc.expert_plan,
        outdir=str(outdir),
        video_name="reach_the_corner_expert_demo.mp4",
    )
    print("Wrote expert demo to {}.".format(video_path))
