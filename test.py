from PIL import Image
import inspect
import sys
import argparse
import json
from fastapi.testclient import TestClient

from utils import logc, bcolors
from detector import Detector

TEST_IMAGE = "images/car2.jpg"
TEST_EXPECTED_LICENSE_PLATE = "CZ20 FSE"

def test_detector():
	_detector = Detector()

	image = Image.open(TEST_IMAGE).convert("RGB")
	image = _detector.preprocess_image(image)
	result = _detector.detect_single(image)

	if result != TEST_EXPECTED_LICENSE_PLATE:
		raise Exception(f"Result {result} | should have been {TEST_EXPECTED_LICENSE_PLATE}")

def test_server():
	from server import app
	client = TestClient(app)
	response = client.post(
		"/detect",
		files={"file": open(TEST_IMAGE, "rb")} 
	).json()

	result = response["license"]

	if result != TEST_EXPECTED_LICENSE_PLATE:
		raise Exception(f"Result {result} | should have been {TEST_EXPECTED_LICENSE_PLATE}")


def run_all_tests():
	test_functions = {
		name: obj for name, obj in inspect.getmembers(sys.modules[__name__]) 
		if (
			inspect.isfunction(obj) 
			and name.startswith('test') 
			and name != 'run_all_tests'
		)
	}

	tests_passed_counter = 0
	tests_failed_counter = 0

	for name, f in test_functions.items():
		print(f"-> Running: [{name}]...")
		
		try:
			f()
			tests_passed_counter += 1
			logc("Passed", bcolors.OKGREEN)
		except Exception as exc:
			tests_failed_counter += 1
			logc(f"Failed: {exc}", bcolors.FAIL)

	print("---")
	print(f"Summary: {tests_passed_counter} passed | {tests_failed_counter} failed")



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--run-all", default=True, action="store_true")
	args = parser.parse_args()

	if args.run_all:
		run_all_tests()
		