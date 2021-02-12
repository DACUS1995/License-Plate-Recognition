import cv2
from PIL import Image
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

from utils import deEmojify

class Detector():
	def __init__(self):
		pass

	def detect_single(self, image):
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		(x1, y1, x2, y2), license_plate_contour = self.localize_plate(gray_image)

		src_pt = np.float32(license_plate_contour)
		dst_pt = np.float32([[x2, y1], [x1, y1], [x1, y2], [x2, y2]])

		M = cv2.getPerspectiveTransform(src_pt, dst_pt)
		gray_image = cv2.warpPerspective(gray_image, M, (gray_image.shape[1], gray_image.shape[0]))

		license_plate = gray_image[y1:y2+1, x1:x2+1]
		license_plate_number = self.extract_characters(license_plate)

		return license_plate_number


	def localize_plate(self, gray_image):
		gray_image = cv2.bilateralFilter(gray_image, 13, 15, 15)

		canny = cv2.Canny(gray_image, 170, 255, 1)
		contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		contours = sorted(contours, key=cv2.contourArea, reverse = True)
		license_plate_contour = None

		for c in contours:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.018 * peri, True)

			if len(approx) == 4:
				license_plate_contour = approx
				cv2.drawContours(gray_image, [license_plate_contour], -1, (0, 0, 255), 3)
				break

		license_plate_contour = np.array(license_plate_contour).squeeze()
		x1 = min(license_plate_contour[:, 0])
		x2 = max(license_plate_contour[:, 0])
		y1 = min(license_plate_contour[:, 1])
		y2 = max(license_plate_contour[:, 1])

		return (x1, y1, x2, y2), license_plate_contour

	def extract_characters(self, plate_image, method="tesseract"):
		if method == "tesseract":
			result = self._apply_tesseract(plate_image)
			return result.split("\n")[0]
		elif method == "custom":
			plate_image = 255 - plate_image
			plate_image[plate_image < 200] = 0
			img_row_sum = np.sum(plate_image,axis=1).tolist()
			plt.plot(img_row_sum)
			plt.show()
			cv2.imshow("image", plate_image)
			cv2.waitKey()
		else:
			raise Exception("Unhandled character extraction method")


	def preprocess_image(self, image):
		if not isinstance(image, np.ndarray):
			image = np.array(image)	

		image = image[:, :, ::-1]
		return image

	def _apply_tesseract(self, plate_image):
		pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'
		text = pytesseract.image_to_string(plate_image, config='--psm 11')
		return text


if __name__ == "__main__":
	det = Detector()

	image = Image.open("car2.jpg").convert("RGB")
	image = np.array(image)

	image = det.preprocess_image(image)
	results = det.detect_single(image)
	print(results)