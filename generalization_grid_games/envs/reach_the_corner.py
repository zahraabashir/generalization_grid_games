from .generalization_grid_game import GeneralizationGridGame, create_gym_envs, InvalidState
from .utils import get_asset_path

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon, FancyArrow
import matplotlib.pyplot as plt
import numpy as np


EMPTY = 'empty'
AGENT = 'agent'
DRAWN = 'drawn'
LEFT_ARROW = 'left_arrow'
RIGHT_ARROW = 'right_arrow'
ALL_TOKENS = [EMPTY, AGENT, DRAWN, LEFT_ARROW, RIGHT_ARROW]


TOKEN_IMAGES = {
    AGENT: plt.imread(get_asset_path('robot.png')),
    DRAWN: plt.imread(get_asset_path('brown_block.jpg')),
}


HAND_ICON_IMAGE = plt.imread(get_asset_path('hand_icon.png'))


class ReachTheCorner(GeneralizationGridGame):

    num_tokens = len(ALL_TOKENS)
    hand_icon = HAND_ICON_IMAGE
    fig_scale = 1.2

    def __init__(self, layout, *args, **kwargs):
        self.goal_position = self.find_goal_position(layout)
        super(ReachTheCorner, self).__init__(layout, *args, **kwargs)

    @staticmethod
    def find_goal_position(layout):
        layout = np.array(layout, dtype=object)
        _, width = layout.shape
        _, agent_c = np.argwhere(layout == AGENT)[0]
        goal_c = width - 1 if agent_c < width / 2.0 else 0
        return (0, goal_c)

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
        agent_position = tuple(np.argwhere(layout == AGENT)[0])
        return agent_position == self.goal_position

    @staticmethod
    def step_move_in_direction(layout, direction):
        _, width = layout.shape

        r, c = np.argwhere(layout == AGENT)[0]

        if c + direction < 0 or c + direction >= width:
            return

        neighbor_cell = layout[r, c + direction]

        if neighbor_cell == EMPTY:
            next_r, next_c = r, c + direction

        elif neighbor_cell == DRAWN and layout[r - 1, c + direction] == EMPTY:
            next_r, next_c = r - 1, c + direction

        else:
            return

        layout[r, c] = EMPTY
        layout[next_r, next_c] = AGENT

    @staticmethod
    def finish_simulation(layout):
        height, width = layout.shape

        while True:
            something_moved = False

            for r in range(height - 2, -1, -1):
                for c in range(width):
                    token = layout[r, c]

                    if (token == AGENT or token == DRAWN) and layout[r + 1, c] == EMPTY:
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
    height = rng.randint(5, 12)
    approach_width = rng.randint(1, 6)
    width = height + approach_width + 2
    layout = np.full((height, width), EMPTY, dtype=object)

    agent_r = height - 3

    if rng.uniform() > 0.5:
        agent_c = rng.randint(0, approach_width + 1)
    else:
        agent_c = width - 1 - rng.randint(0, approach_width + 1)

    layout[agent_r, agent_c] = AGENT
    layout[-2:, :] = DRAWN
    layout[-1, -1] = RIGHT_ARROW
    layout[-1, -2] = LEFT_ARROW

    return layout


layouts = [create_random_layout() for _ in range(num_layouts)]
create_gym_envs(ReachTheCorner, layouts, globals())


def expert_plan(layout):
    layout = np.array(layout, dtype=object)
    height, width = layout.shape
    agent_r, agent_c = np.argwhere(layout == AGENT)[0]
    goal_r, goal_c = ReachTheCorner.find_goal_position(layout)
    move_direction = 1 if goal_c > agent_c else -1
    vertical_steps = agent_r - goal_r

    outer_stair_c = goal_c - move_direction * (vertical_steps - 1)
    before_stair_c = outer_stair_c - move_direction

    if not 0 <= before_stair_c < width:
        raise InvalidState(
            "ReachTheCorner expert needs more horizontal space to build a staircase."
        )

    actions = []
    temp_layout = layout.copy()
    env = ReachTheCorner(layout)

    for step_from_corner in range(vertical_steps):
        c = goal_c - move_direction * step_from_corner
        top_block_r = goal_r + 1 + step_from_corner

        for _ in range(height - 2 - top_block_r):
            action = (top_block_r, c)
            actions.append(action)
            temp_layout = env.transition(temp_layout, action)

    arrow = RIGHT_ARROW if move_direction == 1 else LEFT_ARROW
    arrow_action = tuple(np.argwhere(temp_layout == arrow)[0])
    actions.extend([arrow_action] * abs(before_stair_c - agent_c))
    actions.extend([arrow_action] * vertical_steps)

    return actions
