def trunc(a, threshold=0):
    return threshold if a < threshold else a


def distance(rectA, rectB, p1, p2):
    a1 = rectA[p1]
    a2 = rectA[p2]
    b1 = rectB[p1]
    b2 = rectB[p2]
    scale_a = a2 - a1
    scale_b = b2 - b1
    combined = max(a2, b2) - min(a1, b1)
    return trunc(combined-scale_a-scale_b)


def divide_by_height(dist, rectA, rectB):
    _, ay1, _, ay2 = rectA
    _, by1, _, by2 = rectB
    avg_height = ((ay2 - ay1) + (by2 - by1)) / 2.0
    return dist / avg_height


def x_by_height(rectA, rectB):
    dist = distance(rectA, rectB, 0, 2)
    return divide_by_height(dist, rectA, rectB)


def y_by_height(rectA, rectB):
    dist = distance(rectA, rectB, 1, 3)
    return divide_by_height(dist, rectA, rectB)
