from generalization_grid_games.envs import climb_to_the_block as ctb
from generalization_grid_games.envs.utils import run_random_agent_demo


def run_interactive_demos():
    ctb.ClimbToTheBlockGymEnv1(interactive=True)
    ctb.ClimbToTheBlockGymEnv2(interactive=True)
    ctb.ClimbToTheBlockGymEnv3(interactive=True)
    ctb.ClimbToTheBlockGymEnv4(interactive=True)
    ctb.ClimbToTheBlockGymEnv5(interactive=True)
    ctb.ClimbToTheBlockGymEnv6(interactive=True)


if __name__ == "__main__":
    run_interactive_demos()
    # run_random_agent_demo(ctb.ClimbToTheBlockGymEnv1)
