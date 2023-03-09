import numpy as np
import cv2

class CoverDescriptor:
	def __init__(self, kpMethod = "SIFT", descMethod = "SIFT"):
		self.kpMethod = kpMethod
		self.descMethod = descMethod

	def describe(self, image):
		detector = cv2.SIFT_create()
		(kps, descs) = detector.detectAndCompute(image, None)

		kps = np.float32([kp.pt for kp in kps])

		return (kps, descs)