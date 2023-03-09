# python search.py --db books.csv --covers covers --query queries/query01.png

from pyimagesearch.coverdescriptor import CoverDescriptor
from pyimagesearch.covermatcher import CoverMatcher
import argparse
import glob
import csv
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True,
	help = "path to the book database")
ap.add_argument("-c", "--covers", required = True,
	help = "path to the directory that contains our book covers")
ap.add_argument("-q", "--query", required = True,
	help = "path to the query book cover")
args = vars(ap.parse_args())

db = {}

for l in csv.reader(open(args["db"])):
	db[l[0]] = l[1:]

cd = CoverDescriptor()
cv = CoverMatcher(cd, glob.glob(args["covers"] + "/*.png"))

queryImage = cv2.imread(args["query"])
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = cd.describe(gray)

results = cv.search(queryKps, queryDescs)

cv2.imshow("Query", queryImage)

if len(results) == 0:
	print ("I could not find a match for that cover!")
	cv2.waitKey(0)

else:
	for (i, (score, coverPath)) in enumerate(results):
		(author, title) = db[coverPath[coverPath.rfind("\\") + 1:]]
		print ("%d. %.2f%% : %s - %s" % (i + 1, score * 100, author, title))

		result = cv2.imread(coverPath)
		cv2.imshow("Result", result)
		cv2.waitKey(0)