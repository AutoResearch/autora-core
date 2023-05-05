import sys

import dill
from func0 import controller

filename = sys.argv[1]
with open(filename, "wb") as file:
    dill.dump(controller, file)
