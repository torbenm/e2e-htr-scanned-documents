import numpy as np
import cv2


def generate_text(text, font, target_height, thickness=1, padding=5):
    font_scale = target_height / 30.0
    size = cv2.getTextSize(text, font, font_scale, thickness)
    line_height = size[0][1]+size[1]
    image = np.zeros((line_height+padding*4, size[0][0]+padding))
    cv2.putText(image, text, (0, size[0][1]+padding), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    print(size)
    return image


fonts = [
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX
    cv2.
]

for idx, font in enumerate(fonts):
    img = generate_text(
        'majority|in|Northern|Rhodesia|,|but|the'.replace('|', ' '), font, 40)
    # img = np.zeros((60, 500, 1), np.uint8)

    cv2.imwrite('output{}.png'.format(idx), img)
