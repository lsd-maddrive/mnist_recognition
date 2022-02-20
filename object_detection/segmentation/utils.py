import cv2
import numpy as np


def choose_threshold(img: np.ndarray) -> np.ndarray:
    """
    Description of the Function
    Function for creating mask for segmentaion.
    It will let the user choose the value of mask parametrs by using the Trackbar.

    Parameters:
    img (np.ndarray): Original image


    Returns:
    mask (np.ndarray): mask for image segmentation based on choosen parametrs
    """

    def TrackbarCallback(pos: int) -> None:
        """
        Callback function for Trackbar.
        Parameters:
        pos(int): Сurrent position of the specified trackbar.
        """

    cv2.namedWindow("Color Track Bar", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("low_hue", "Color Track Bar", 0, 255, TrackbarCallback)
    cv2.createTrackbar("low_sat", "Color Track Bar", 0, 255, TrackbarCallback)
    cv2.createTrackbar("low_val", "Color Track Bar", 0, 255, TrackbarCallback)
    cv2.createTrackbar("high_hue", "Color Track Bar", 0, 255, TrackbarCallback)
    cv2.createTrackbar("high_sat", "Color Track Bar", 0, 255, TrackbarCallback)
    cv2.createTrackbar("high_val", "Color Track Bar", 0, 255, TrackbarCallback)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    while True:
        low_hue = cv2.getTrackbarPos("low_hue", "Color Track Bar")
        low_sat = cv2.getTrackbarPos("low_sat", "Color Track Bar")
        low_val = cv2.getTrackbarPos("low_val", "Color Track Bar")
        high_hue = cv2.getTrackbarPos("high_hue", "Color Track Bar")
        high_sat = cv2.getTrackbarPos("high_sat", "Color Track Bar")
        high_val = cv2.getTrackbarPos("high_val", "Color Track Bar")

        low_color = (low_hue, low_sat, low_val)
        high_color = (high_hue, high_sat, high_val)

        mask = cv2.inRange(img_hsv, low_color, high_color)
        result = cv2.bitwise_and(img, img, mask=mask)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        cv2.imshow("Mask", mask)

        cv2.imshow("Masked img", result_bgr)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    return mask
