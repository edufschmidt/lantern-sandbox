import os
import sys

# Append parent directory to the path so that we can access
#  our module from the scripts in the /bin directory.

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))