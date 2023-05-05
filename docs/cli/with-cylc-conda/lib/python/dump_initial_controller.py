import sys

import dill
from controller_setup import controller

filename = sys.argv[1]
with open(filename, "wb") as file:
    dill.dump(controller, file)
