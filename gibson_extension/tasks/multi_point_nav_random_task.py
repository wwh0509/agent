from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.utils.utils import l2_distance
import pybullet as p
import logging
import numpy as np
import json
import os


class MultiPointRandomNavTask(PointNavFixedTask):
    """
    Point Nav Random Task
    The goal is to navigate to a random goal position
    """

    def __init__(self, env):
        super(MultiPointRandomNavTask, self).__init__(env)
        self.target_dist_min = self.config.get('target_dist_min', 1.0)
        self.target_dist_max = self.config.get('target_dist_max', 10.0)
        self.num_point = self.config.get('num_point', 2)
        self.test = self.config.get('test', False)
        if self.test:
            self.episode_data = json.load(open(env.config['scene_id'] + '.json', 'r'))
            self.total_episodes = len(self.episode_data)

    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        max_trials = 100
        dist = 0.0
        self.target_pos_list = []
        for _ in range(self.num_point):
            for _ in range(max_trials):
                _, target_pos = env.scene.get_random_point(floor=self.floor_num)
                if env.scene.build_graph:
                    _, dist = env.scene.get_shortest_path(
                        self.floor_num,
                        initial_pos[:2],
                        target_pos[:2], entire_path=False)
                else:
                    dist = l2_distance(initial_pos, target_pos)
                if self.target_dist_min < dist < self.target_dist_max:
                    break
            if not (self.target_dist_min < dist < self.target_dist_max):
                print("WARNING: Failed to sample initial and target positions")
            self.target_pos_list.append(target_pos)
            
        self.target_pos_list = np.array(self.target_pos_list).reshape(-1, 3)
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn, self.target_pos_list

    def reset_scene(self, env):
        """
        Task-specific scene reset: get a random floor number first

        :param env: environment instance
        """
        self.floor_num = env.scene.get_random_floor()
        super(MultiPointRandomNavTask, self).reset_scene(env)

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn, target_pos_list = \
                self.sample_initial_pose_and_target_pos(env)
            reset_success = env.test_valid_position(
                env.robots[0], initial_pos, initial_orn)
            for j in range(self.num_point):
                reset_success = reset_success and env.test_valid_position(
                    env.robots[0], target_pos_list[j])
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        # removed cached state to prevent memory leak
        p.removeState(state_id)

        if not self.test:
            self.target_pos = target_pos_list[0]
            self.initial_pos = initial_pos
            self.initial_orn = initial_orn
        else:
            # 根据env.current_episode读取episode_data的数据
            episode_idx = str(env.current_episode + 1)
            self.target_pos = np.array(self.episode_data[episode_idx][0])
            self.initial_pos = np.array(self.episode_data[episode_idx][1])
            self.initial_orn = np.array(self.episode_data[episode_idx][2])

        super(MultiPointRandomNavTask, self).reset_agent(env)
