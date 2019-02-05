import cv2

from lib.segmentation.AStarLineSegmentation import AStarLineSegmentation

if __name__ == "__main__":

    img = cv2.imread("research/astar/ex02.png", cv2.IMREAD_GRAYSCALE)
    lineseg = AStarLineSegmentation()
    lines, _ = lineseg(img)
    for line in lines:
        print(line)
        cv2.imshow('line', line.img)
        cv2.waitKey(0)
