import numpy as np

from random import shuffle
from datasets import load_dataset

from ...base import BaseCourse
from ...core import StateDescription, Action


class MMLUnaire(BaseCourse):
    """
    Course to teach an an agent multiple-choice
    """

    def __init__(
        self,
        num_questions: int = 10,
        seed: int = 1,
        subdata: str|None = None,
        test_or_val: str = 'test'
    ):
        super(MMLUnaire, self).__init__()

        # The num_questions acts as the horizon
        self.num_questions = num_questions
        self.seed = seed
        self.subdata = subdata
        self.test_or_val = test_or_val

        self.dataset = load_dataset(
            "cais/mmlu",
            subdata if subdata is not None else "all"
        )

        self.rewards = np.arange(self.num_questions)

    def reset(self):
        # Each round start with the agent at $0, and on the first tier
        self.tier = 1
        self.money = 0
        self.idx = 0

        question, choices = self._generate_questions()

        description = f"""
        You are playing a game where you must choose the correct answer out of the possible choices.
        Your goal is to pick the correct choice each time to make as much money as possible.
        Score: {self.money}

        Question:
        {question}

        Choices:
        {choices}
        """
        return StateDescription(description)

    def step(self, action: Action) -> tuple[StateDescription, int, bool, bool, dict[str, any]]:
        # the answer is a type of int, the index of the correct choice
        # if self.idx
        if self.tier == 1:
            answer = self


    def _generate_questions(self):
        dataset = load_dataset(
            "cais/mmlu",
            self.subdata if self.subdata is not None else "all"
        )
        if 'test' in self.test_or_eval:
            question_set = dataset['test'].shuffle(seed=self.seed)
        elif 'val' in self.test_or_eval:
            question_set = dataset['val'].shuffle(seed=self.seed)
        else:
            raise ValueError(f"test_or_val must be 'test' or 'val'. Got {self.test_or_val}")
            
        question_set_size = question_set.num_rows
        questions_per_tier = question_set_size//3

        self.first_tier = question_set[:questions_per_tier]
        self.second_tier = question_set[questions_per_tier:2*questions_per_tier]
        self.third_tier = question_set[2*questions_per_tier:]

        question = self.first_tier['question'][0]
        choices = shuffle(self.first_tier['choices'][0])
        return question, choices
        
