import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--nary", type=int,
                    action="store", nargs="*")

parser.add_argument("--nary_tup", type=int,
                    action="append", nargs="+")

known, unknown = parser.parse_known_args()

print(vars(known))