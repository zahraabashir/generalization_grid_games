from generalization_grid_games.envs import reach_the_corner as rtc
from generalization_grid_games.envs.utils import run_random_agent_demo


def run_interactive_demos():
    rtc.ReachTheCornerGymEnv1(interactive=True)
    rtc.ReachTheCornerGymEnv2(interactive=True)
    rtc.ReachTheCornerGymEnv3(interactive=True)
    rtc.ReachTheCornerGymEnv4(interactive=True)
    rtc.ReachTheCornerGymEnv5(interactive=True)
    rtc.ReachTheCornerGymEnv6(interactive=True)


if __name__ == "__main__":
    run_interactive_demos()
    # run_random_agent_demo(rtc.ReachTheCornerGymEnv1)
