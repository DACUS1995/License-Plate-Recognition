from PIL import Image
import inspect
import sys
import argparse

from utils import logc, bcolors
from detector import Detector

def test_detector():
	_detector = Detector()

	image = Image.open("car2.jpg").convert("RGB")
	image = _detector.preprocess_image(image)
	result = _detector.detect_single(image)

	expected = "CZ20 FSE"

	if result != expected:
		raise Exception(f"Result {result} | should have been {expected}")

def test_server():
	pass


def run_all_tests():
	test_functions = {
		name: obj for name, obj in inspect.getmembers(sys.modules[__name__]) 
		if (
			inspect.isfunction(obj) 
			and name.startswith('test') 
			and name != 'run_all_tests'
		)
	}

	for name, f in test_functions.items():
		print(f"-> Running: [{name}]...")
		
		try:
			f()
			logc("Passed", bcolors.OKGREEN)
		except Exception as exc:
			logc(f"Failed: {exc}", bcolors.FAIL)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--run-all", default=True, action="store_true")
	args = parser.parse_args()

	if args.run_all:
		run_all_tests()
		