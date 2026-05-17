from generalization_grid_games.envs import climb_to_the_block as ctb
from generalization_grid_games.envs.utils import run_plan_demo
from pathlib import Path


if __name__ == "__main__":
    outdir = Path(__file__).resolve().parent / "out"
    video_path = run_plan_demo(
        ctb.ClimbToTheBlockGymEnv3,
        ctb.expert_plan,
        outdir=str(outdir),
        video_name="expert_demo.mp4",
    )
    print("Wrote expert demo to {}.".format(video_path))
