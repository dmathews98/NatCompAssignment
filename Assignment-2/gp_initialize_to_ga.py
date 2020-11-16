from gp import *
from main import *

def neworg(self):
    to_return = GPTree(
        label='ReLU',
        info=5,
        info2=2,
        children=[
            GPTree(
                label='ReLU',
                info=2,
                info2=1,
                children=[
                    GPTree(
                        label='ReLU',
                        info=7,
                        info2=0,
                        children=[
                            GPTree(
                                label='ReLU',
                                info=4,
                                info2=3,
                                children=[
                                    GPTree('Input'),
                                    GPTree('Input')
                                ]
                            ),
                            GPTree('Input')
                        ]
                    ),
                    GPTree('Input')
                ]
            ),
            GPTree('Input')
        ]
    )
    return to_return

GP.starting_organism = lambda self: neworg(self)
run_script('Assignment-2/insanity_check.txt')