from .generalization_grid_game import GeneralizationGridGame, create_gym_envs, InvalidState
from .utils import get_asset_path

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon, FancyArrow
import matplotlib.pyplot as plt
import numpy as np


EMPTY = 'empty'
AGENT = 'agent'
STAR = 'star'
DRAWN = 'drawn'
LEFT_ARROW = 'left_arrow'
RIGHT_ARROW = 'right_arrow'
ALL_TOKENS = [EMPTY, AGENT, DRAWN, STAR, LEFT_ARROW, RIGHT_ARROW]


TOKEN_IMAGES = {
    AGENT: plt.imread(get_asset_path('robot.png')),
    DRAWN: plt.imread(get_asset_path('brown_block.jpg')),
    STAR: plt.imread(get_asset_path('star.png')),
}


HAND_ICON_IMAGE = plt.imread(get_asset_path('hand_icon.png'))


class ClimbToTheBlock(GeneralizationGridGame):

    num_tokens = len(ALL_TOKENS)
    hand_icon = HAND_ICON_IMAGE
    fig_scale = 1.2

    def __init__(self, layout, *args, **kwargs):
        self.goal_position = self.find_goal_position(layout)
        super(ClimbToTheBlock, self).__init__(layout, *args, **kwargs)

    @staticmethod
    def find_goal_position(layout):
        layout = np.array(layout, dtype=object)
        height, _ = layout.shape
        candidate_positions = np.argwhere(layout == DRAWN)

        goal_positions = [
            (r, c) for r, c in candidate_positions
            if r < height - 2
        ]

        if len(goal_positions) != 1:
            raise InvalidState(
                "ClimbToTheBlock expects exactly one non-ground DRAWN cell to act as the goal."
            )

        return goal_positions[0]

    def transition(self, layout, action):
        r, c = action
        token = layout[r, c]
        new_layout = layout.copy()

        if token == EMPTY:
            new_layout[r, c] = DRAWN

        elif token == LEFT_ARROW:
            self.step_move_in_direction(new_layout, -1)

        elif token == RIGHT_ARROW:
            self.step_move_in_direction(new_layout, 1)

        else:
            return new_layout

        self.finish_simulation(new_layout)

        return new_layout

    def compute_reward(self, layout0, action, layout1):
        return float(self.compute_done(layout1))

    def compute_done(self, layout):
        agent_r, agent_c = np.argwhere(layout == AGENT)[0]
        goal_r, goal_c = self.goal_position
        return (agent_r == goal_r - 1) and (agent_c == goal_c)

    @staticmethod
    def step_move_in_direction(layout, direction):
        height, width = layout.shape

        r, c = np.argwhere(layout == AGENT)[0]

        if c + direction < 0 or c + direction >= width:
            return

        neighbor_cell = layout[r, c + direction]

        if neighbor_cell in [EMPTY, STAR]:
            next_r, next_c = r, c + direction

        elif neighbor_cell == DRAWN and layout[r - 1, c + direction] in [EMPTY, STAR]:
            next_r, next_c = r - 1, c + direction

        else:
            return

        layout[r, c] = EMPTY
        layout[next_r, next_c] = AGENT

    def finish_simulation(self, layout):
        height, width = layout.shape

        while True:
            something_moved = False

            for r in range(height - 2, -1, -1):
                for c in range(width):
                    token = layout[r, c]

                    if (r, c) == self.goal_position:
                        continue

                    if (token == AGENT or token == DRAWN) and (layout[r + 1, c] == EMPTY):
                        layout[r, c] = EMPTY
                        layout[r + 1, c] = token
                        something_moved = True

            if not something_moved:
                break

    @classmethod
    def draw_token(cls, token, r, c, ax, height, width, token_scale=1.0):
        if token == EMPTY:
            return None

        if 'arrow' in token:
            edge_color = '#888888'
            face_color = '#AAAAAA'

            drawing = RegularPolygon(
                (c + 0.5, (height - 1 - r) + 0.5),
                numVertices=4,
                radius=0.5 * np.sqrt(2),
                orientation=np.pi / 4,
                ec=edge_color,
                fc=face_color,
            )
            ax.add_patch(drawing)

        if token == LEFT_ARROW:
            arrow_drawing = FancyArrow(
                c + 0.75,
                height - 1 - r + 0.5,
                -0.25,
                0.0,
                width=0.1,
                fc='green',
                head_length=0.2,
            )
            ax.add_patch(arrow_drawing)

        elif token == RIGHT_ARROW:
            arrow_drawing = FancyArrow(
                c + 0.25,
                height - 1 - r + 0.5,
                0.25,
                0.0,
                width=0.1,
                fc='green',
                head_length=0.2,
            )
            ax.add_patch(arrow_drawing)

        else:
            im = TOKEN_IMAGES[token]
            oi = OffsetImage(im, zoom=cls.fig_scale * (token_scale / max(height, width) ** 0.5))
            box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)

            ax.add_artist(box)

            return box

    @classmethod
    def initialize_figure(cls, height, width):
        fig, ax = GeneralizationGridGame.initialize_figure(height, width)

        for r in range(height):
            for c in range(width):
                edge_color = '#888888'
                face_color = 'white'

                drawing = RegularPolygon(
                    (c + 0.5, (height - 1 - r) + 0.5),
                    numVertices=4,
                    radius=0.5 * np.sqrt(2),
                    orientation=np.pi / 4,
                    ec=edge_color,
                    fc=face_color,
                )
                ax.add_patch(drawing)

        return fig, ax


### Specific environments
rng = np.random.RandomState(0)
num_layouts = 20


def create_random_layout():
    stairs_height = rng.randint(2, 11)
    stairs_dist_from_right = rng.randint(0, 5)
    stairs_dist_from_top = rng.randint(0, 5)
    agent_dist_from_stairs = rng.randint(0, 5)
    agent_dist_from_left = rng.randint(0, 5)

    height = 2 + stairs_height + stairs_dist_from_top
    width = stairs_dist_from_right + 2 * stairs_height + agent_dist_from_stairs + agent_dist_from_left
    layout = np.full((height, width), EMPTY, dtype=object)

    goal_r = stairs_dist_from_top + 1
    goal_c = agent_dist_from_left + agent_dist_from_stairs + stairs_height
    agent_r = height - 3
    agent_c = agent_dist_from_left

    star_r = goal_r - 1
    star_c = width - 1 - goal_c
    if star_r == goal_r and star_c == goal_c:
        star_c = max(0, star_c - 1)

    layout[goal_r, goal_c] = DRAWN
    layout[star_r, star_c] = STAR
    layout[agent_r, agent_c] = AGENT

    if rng.uniform() > 0.5:
        layout = np.fliplr(layout)

    layout[-2:, :] = DRAWN
    layout[-1, -1] = RIGHT_ARROW
    layout[-1, -2] = LEFT_ARROW

    return layout


layouts = [create_random_layout() for _ in range(num_layouts)]
create_gym_envs(ClimbToTheBlock, layouts, globals())
