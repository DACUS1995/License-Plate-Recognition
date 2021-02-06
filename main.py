import argparse

from server import app

def main(args):
	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--debug-mode-enabled", default=False, action="store_true")
	args = parser.parse_args()
	main(args)