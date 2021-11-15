import cv2
import numpy as np

from typing import List


def image_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses the image to get a better line detection
    Args:
        image: image to process

    Returns:
        result: processed image
    """
    blur = cv2.blur(image, (5, 5))

    _, thresh1 = cv2.threshold(blur, 168, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(thresh1, cv2.COLOR_RGB2HSV)

    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel_erode = np.ones((6, 6), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((4, 4), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)

    return dilated_mask


def direction(image: np.ndarray) -> float:
    """
    Reads and preprocesses the image in path
    Returns a direction between 0 and width in pixels
    The direction is the center of the main detected line

    Args:
        image: image to analyze

    Returns:
        cx: center of the main line detected in pixels
    """

    _, width = image.shape[:2]

    # Preprocess image (blur, threshold, etc.)
    dilated_mask = image_preprocessing(image)

    # Find contours
    contours, _ = cv2.findContours(
        dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If contours found, take the biggest one
    if (len(contours) > 0):
        main_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        coords = cv2.moments(main_contour)
        if (coords['m00'] != 0):
            cx = int(coords['m10']/coords['m00'])
            # Return its center
            return cx
    # If no contours found, return the center of the image
    return width/2


def turn_detection(image: np.ndarray, spacing_ratio: float = 1/4, surface_ratio: float = 0.01,
                   center_surface_ratio: float = 0.05) -> List[bool]:
    """
    Reads and preprocesses the image in path
    Returns two booleans [x,y]
    x is True if there is a left turn, False otherwise
    y is the same for a right turn

    Args:
        image: image to analyze
        spacing_ratio: if the line detected width*spacing_ratio pixels from the border,
            there is a left turn, same for right
        surface_ratio: if (area of the line)/(area of the image) is more than surface_ratio,
            there is a turn

    Returns:
        two bools describing the presence of a left/right turn
    """
    h, w = image.shape[:2]

    # Isolate left and right side of the image
    # Crop image vertically (we want to detect intersection only if it is close)
    left_image = image[int(h/2):, :int(w*spacing_ratio)]
    right_image = image[int(h/2):, int(w*(1 - spacing_ratio)):]
    center_image = image[:, int(w*spacing_ratio):int(w*(1 - spacing_ratio))]

    images = [left_image, right_image]
    result = [False, False]
    # center_lane checks if there is a lane in the middle
    # if there is not, we are not at an intersection
    center_lane = False

    # Preprocess the image (blur, threshold...)
    dilated_mask = image_preprocessing(center_image)

    # Find contours, detection of center lane
    contours, _ = cv2.findContours(
        dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # If contour found:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        # Get the area of the biggest contour
        area = cv2.contourArea(contours[0])
        # Compare to surface ratio
        if area/(w*h) > center_surface_ratio:
            center_lane = True

    # Detection of side lanes
    for i, side_image in enumerate(images):
        # Preprocessing
        dilated_mask = image_preprocessing(side_image)

        # Find contours
        contours, _ = cv2.findContours(
            dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # If contour found:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
            # Get the area of the biggest contour
            area = cv2.contourArea(contours[0])
            # Compare to surface ratio
            if area/(w*h) > surface_ratio and center_lane:
                result[i] = True
    return result


class PID():
    def __init__(self, KP: float, KI: float, KD: float, saturation_max: float, saturation_min: float):
        """
        Params:
            KP: Proportional gain
            KI: Integral gain
            KD: Derivative gain
            saturation_max: maximum value of the output
            saturation_min: minimum value of the output
        """
        self.kp = KP
        self.ki = KI
        self.kd = KD
        self.error_last = 0
        self.integral_error = 0
        self.saturation_max = saturation_max
        self.saturation_min = saturation_min

    def compute(self, error: float, dt: float = 0.1) -> float:
        """
        Compute PID output according to error and dt

        Params:
            error: error to correct
            dt: time step

        Returns:
            output of PID
        """
        derivative_error = (error - self.error_last) / dt
        self.integral_error += error * dt
        output = self.kp*error + self.ki*self.integral_error + self.kd*derivative_error
        self.error_last = error
        if output > self.saturation_max and self.saturation_max is not None:
            output = self.saturation_max
        elif output < self.saturation_min and self.saturation_min is not None:
            output = self.saturation_min
        return output


def direction_to_motor(cx: int, pid: PID, image_width: int = 192, dt: float = 0.1) -> List[float]:
    """
    Converts cx to values to send to the motors using a PID controller
    cx is a value between 0 and image_width
    returns two values between -100 and 100 to feed the left and right motors

    Args:
        cx: center of the main line detected in pixels
        pid: PID controller
        image_width: width of the image in pixels
        dt: time between two frames

    Returns:
        values to send to the motors
    """
    result = [100, 100]
    centered_cx = cx - image_width/2
    pid_output = pid.compute(-centered_cx, dt)

    if pid_output > 0:
        result[0] -= abs(pid_output)
    else:
        result[1] -= abs(pid_output)

    return result


def roundabout_direction(image: np.ndarray,
                         padding_ratio: int = 0.4,
                         h_crop: float = 1/3,
                         v_crop: float = 0,
                         current_lane: str = "Left") -> float:
    """
    Reads and preprocesses the image in path
    Returns direction in pixels (float) so that
    the robot moves right or left of the line detected
    Useful for traversing roundabouts

    Args:
        image: image to process
        padding_ratio: move the output direction by this amount*width
        h_crop: crop the image by this ratio horizontally
        v_crop: crop the image by this ratio vertically
        current_lane: current lane followed by the robot

    Returns:
        cx: direction order for the robot, between 0 and width
    """

    # Load image
    height, width = image.shape[:2]

    # Crop the image to the left bottom corner
    if current_lane == "Left":
        h_crop_size = int(height * (1-h_crop))
        v_crop_size = int(width * v_crop)
        image = image[v_crop_size:, :h_crop_size]
    # Crop the image to the right bottom corner
    else:
        h_crop_size = int(height * h_crop)
        v_crop_size = int(width * v_crop)
        image = image[v_crop_size:, h_crop_size:]

    # Preprocess image (blur, threshold, etc.)
    dilated_mask = image_preprocessing(image)

    # Find contours
    contours, _ = cv2.findContours(
        dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if contours found, take the biggest one
    if len(contours) > 0:
        main_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        coords = cv2.moments(main_contour)
        if coords['m00'] != 0:
            cx = int(coords['m10']/coords['m00'])
            if current_lane == "Left":
                return cx + padding_ratio*width
            else:
                return cx + h_crop_size - padding_ratio*width

    if current_lane == "Right":
        return width/2
    else:
        return width/3


def detect_red(image: np.ndarray) -> bool:
    """
    Detects if there is a red color in the image
    Args:
        image: image to process

    Returns:
        bool: True if red is detected
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV (could differ depending on lighting)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Erode and dilate the mask
    kernel_erode = np.ones((6, 6), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((4, 4), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if contours found, take the biggest one
    if len(contours) > 0:
        main_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        coords = cv2.moments(main_contour)
        if coords['m00'] != 0:
            _ = int(coords['m10']/coords['m00'])
            _ = int(coords['m01']/coords['m00'])
            return True
    return False
