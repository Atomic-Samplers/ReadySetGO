from quansino.mc import Canonical
from .random import Random


rand_atoms= Random().go_suggest()


mc = Canonical(
    atoms=None,
    temperature=300,
    num_cycles=1000,
    default_move=None,
    mc_kwargs={},
)


# class BasinHopping():
#     def __init__():
#         pass
#         return