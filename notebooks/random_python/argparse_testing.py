# 09/20/22 - testing input for multiple cli args for wavelength bounds
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--wvb', type=int, nargs='+', default=None)
parser.add_argument("--str", type=str, default=None)
args = parser.parse_args()
k_args = parser.parse_known_args()
print(args)
print(k_args)