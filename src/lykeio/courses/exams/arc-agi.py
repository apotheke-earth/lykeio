import numpy as np

from random import shuffle
from alectors import Agent
from datasets import load_dataset

from ...base import BaseCourse
from ...core import StateDescription, Action


class ARCAGI(BaseCourse):
    """
    RL Env for the ARC-AGI-2 competition
    """
    def __init__(
        self,
        seed: int = 1,
        is_eval: bool = False
    ):
        super(ARCAGI, self).__init__()

        self.dataset = load_dataset(
            'arc-agi-community/arc-agi-2',
        )


    def reset(self) -> StateDescription:
        self.idx = 0
        self.step_to_horizon = 0
        self.action_class = 'column'
        if not is_eval:
            self.exam_set = self.dataset['train'].shuffle(seed=self.seed)
        else:
            self.exam_set = self.dataset['test'].shuffle(seed=self.seed)

        self._fewshots = self.exam_set['fewshots'][self.idx]
        self._questions = self.exam_set['question'][self.idx]

        self.row_size = len(self._question[0][0]['output'])
        self.col_size = len(self._question[0][0]['output'][0])

        self.output_grid = np.zeros(shape=(self.row_size, self.col_size))

        description = f"""
        You are partaking in an exam.
        Your task is to guess the correct pattern, based on the examples.
        The examples are:
        {self._fewshots}
        The input for the question is:
        {self._question[0][0]['input']}
        The size of the output is:
        ({len(self._question[0][0]['output'])}, {self._question[0][0]['output'][0]})
        you are currently picking the {self.action_class}
        """


    def step(
        self,
        action: Action
    ) -> tuple[StateDescription, int, bool, bool, dict[str, any]]:

        if 'row' in self.action_class:
            row = action
            if row >= self.row_size:
                reward = -100
            else:    
                reward = 0
            self.action_class = 'color'

        elif 'column' in self.action_class:
            column = action
            if column >= self.col_size:
                reward = -100
            else:
                reward = 0
            self.action_class = 'row'


        elif 'color' in self.action_class:
            if action >= 10:
                reward = -100
            else:
                self.output_grid[self.column][self.row] = action
                reward = self._calc_reward(action)

            self.action_class = 'column'
                
        self._fewshots = self.exam_set['fewshots'][self.idx]
        self._questions = self.exam_set['question'][self.idx]

        description = f"""
        You are partaking in an exam.
        Your task is to guess the correct pattern, based on the examples.
        The examples are:
        {self._fewshots}
        The input for the question is:
        {self._question[0][0]['input']}
        The size of the output is:
        ({self.row_size}, {self.col_size}).
        So far your solution is:
        {self.output_grid}
        You are currently selecting the {self.action_class}
        """

    def available_actions(self) -> list[Action]:
        """
        Returns all available actions: 'pick_item_(1-30)'.
        """
        action_list = [
            Action(action='pick_item_1', description="Picks the available col/row/color"),
            Action(action='pick_item_2', description="Picks the available col/row/color"),
            Action(action='pick_item_3', description="Picks the available col/row/color"),
            Action(action='pick_item_4', description="Picks the available col/row/color"),
            Action(action='pick_item_5', description="Picks the available col/row/color"),
            Action(action='pick_item_6', description="Picks the available col/row/color"),
            Action(action='pick_item_7', description="Picks the available col/row/color"),
            Action(action='pick_item_8', description="Picks the available col/row/color"),
            Action(action='pick_item_9', description="Picks the available col/row/color"),
            Action(action='pick_item_10', description="Picks the available col/row/color"),
            Action(action='pick_item_11', description="Picks the available col/row/color"),
            Action(action='pick_item_12', description="Picks the available col/row/color"),
            Action(action='pick_item_13', description="Picks the available col/row/color"),
            Action(action='pick_item_14', description="Picks the available col/row/color"),
            Action(action='pick_item_15', description="Picks the available col/row/color"),
            Action(action='pick_item_16', description="Picks the available col/row/color"),
            Action(action='pick_item_17', description="Picks the available col/row/color"),
            Action(action='pick_item_18', description="Picks the available col/row/color"),
            Action(action='pick_item_19', description="Picks the available col/row/color"),
            Action(action='pick_item_20', description="Picks the available col/row/color"),
            Action(action='pick_item_21', description="Picks the available col/row/color"),
            Action(action='pick_item_22', description="Picks the available col/row/color"),
            Action(action='pick_item_23', description="Picks the available col/row/color"),
            Action(action='pick_item_24', description="Picks the available col/row/color"),
            Action(action='pick_item_25', description="Picks the available col/row/color"),
            Action(action='pick_item_26', description="Picks the available col/row/color"),
            Action(action='pick_item_27', description="Picks the available col/row/color"),
            Action(action='pick_item_28', description="Picks the available col/row/color"),
            Action(action='pick_item_29', description="Picks the available col/row/color"),
            Action(action='pick_item_30', description="Picks the available col/row/color"),
        ]
        return action_list

    def _calc_reward(self, action) -> int:
        solution = self._questions[0][0]['output']
        reward = 0
        for i in range(self.row_size):
            for j in range(self.col_size):
                if self.output_grid[i][j] == solution[i][j]:
                    if solution[i][j] != 0:
                        reward += 1
                else:
                    if solution[i][j] != 0:
                        reward -= .3
                    else:
                        reward -= .7
                    
        return reward

                

