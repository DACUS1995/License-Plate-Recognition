from PIL import Image
import inspect
import sys
import argparse

from detector import Detector

def test_detector():
	_detector = Detector()

	image = Image.open("car.jpg").resize((800,800)).convert("RGB")
	image = _detector.preprocess_image(image)

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
			print(f"Passed")
		except Exception as exc:
			print(f"Failed: {exc}")



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--run-all", default=False, action="store_true")
	args = parser.parse_args()

	if args.run_all:
		run_all_tests()
		