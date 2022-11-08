import numpy as np
import sys
np.set_printoptions(threshold=np.inf)

data = np.load(sys.argv[1])
print(data)