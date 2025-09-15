import gym
import generalization_grid_games

if __name__=="__main__":

    for base_class_name in ["TwoPileNim", "CheckmateTactic", "Chase", "StopTheFall", "ReachForTheStar"]:
        for task_instance in range(20):
            env_name = "{}{}-v0".format(base_class_name, task_instance)
            env = gym.make(env_name)
            obs = env.reset()
            for _ in range(50):
                action = env.action_space.sample()
                obs, reward, done, debug_info = env.step(action)
                if done:
                    break
            print("Done")
