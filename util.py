import math

class Box:
    def __init__(self, x, y, w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class Image:
    def __init__(self, name, faces_info):
        self.name = name
        self.bounding_box = []

        for i in range(len(faces_info)):
            self.bounding_box.append(Box(int(faces_info[i][0]), int(faces_info[i][1]), int(faces_info[i][2]), int(faces_info[i][3])))

def common_area(xmin1, ymin1, w1, h1, xmin2, ymin2, w2, h2):   # returns None if rectangles don't intersect
	xmax1 = xmin1 + w1
	ymax1 = ymin1 + h1
	xmax2 = xmin2 + w2
	ymax2 = ymin2 + h2
	dx = min(xmax1, xmax2) - max(xmin1, xmin2)
	dy = min(ymax1, ymax2) - max(ymin1, ymin2)
	if (dx>=0) and (dy>=0):
		return dx*dy
	else:
		return 0

def total_area(xmin1, ymin1, w1, h1, xmin2, ymin2, w2, h2):
	com_area = common_area(xmin1, ymin1, w1, h1, xmin2, ymin2, w2, h2)
	sum = (w1*h1) + (w2*h2)
	return sum - com_area

def accuracy(xmin1, ymin1, w1, h1, xmin2, ymin2, w2, h2):
    return (common_area(xmin1, ymin1, w1, h1, xmin2, ymin2, w2, h2) / total_area(xmin1, ymin1, w1, h1, xmin2, ymin2, w2, h2)) * 100
# print("area" + str(total_area(0, 0, 2, 2, 1, 1, 2, 2)))