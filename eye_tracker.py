import numpy as np
import dlib
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # http://dlib.net/face_landmark_detection.py.html

LEFT_EYE = (42, 48)
RIGHT_EYE = (36, 42)


def select_face(faces: np.ndarray) -> np.ndarray:
    """ 
    Selects the biggest face of 'faces'.
  
    Chooses the one with the biggest area.
  
    Parameters: 
    faces (np.ndarray): Array with the faces.
  
    Returns: 
    np.ndarray: Biggest face.
  
    """
    idx = 0
    
    if len(faces) > 1:
        for i in range(len(faces)):
            if (faces[idx][2] * faces[idx][3]) < (faces[i][2] * faces[i][3]):
                idx = i

    return faces[idx]


def detect_face(frame: np.ndarray) -> dlib.rectangle:
    """ 
    Detects faces in a given frame and returns the coords
    of the face chose by 'select_face'.
  
    Parameters: 
    frame (np.ndarray): wonder what this might be...
  
    Returns: 
    dlib.rectangle: Coords of the face.
  
    """
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    if faces == (): return None  

    x, y, w, h = select_face(faces)

    return dlib.rectangle(x, y, x+w, y+h)


def shape_to_np(shape) -> np.ndarray:
    """ 
    Converts dlib's shape into a numpy array.
  
    Parameters: 
    shape: dlib's shape.
  
    Returns: 
    np.ndarray: Numpy array representing the shape.
  
    """
    array = []

    for point in shape.parts():
        array.append([point.x, point.y])

    return np.array(array)


def get_eyes_keypoints(shape: np.ndarray) -> list:
    """ 
    Returns the points surrounding each eye.
  
    Parameters: 
    shape (np.ndarray): Face landmark keypoints. 
  
    Returns: 
    list: Points of the left and right eye respectively.
  
    """
    li, lf = LEFT_EYE
    left_eye = shape[li:lf]

    ri, rf = RIGHT_EYE
    right_eye = shape[ri:rf]

    return np.array([left_eye]), np.array([right_eye])


def crop_eye(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """ 
    Crops the eye off the frame given some points. 

    Parameters: 
    frame (np.ndarray): Just a frame...
    points (np.ndarray): Points surrounding the eye.
  
    Returns: 
    np.ndarray: Cropped eye.
  
    """
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, points, (255)) 

    res = cv2.bitwise_and(frame, frame, mask=mask) 

    x, y, w, h = cv2.boundingRect(points)

    white_bg = np.ones_like(frame, np.uint8) * 255 
    
    res2 = white_bg + res #!

    cropped = res2[y:y+h, x:x + w]

    return cropped


def preprocess_eye(eye: np.ndarray, threshold: float = 100) -> np.ndarray:
    """ 
    Applies filters to the given image in order to get a blob. 
  
    Parameters: 
    eye (np.ndarray): Grayscaled image of an eye.
    threshold (float): Value used to classify every pixel value. See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#simple-thresholding
  
    Returns: 
    np.ndarray: Preprocess image of the eye.
  
    """
    # Reduce noise
    eye = cv2.GaussianBlur(eye, (7, 7), 0)

    # Apply theshold
    _, eye = cv2.threshold(eye, threshold, 255, cv2.THRESH_BINARY)

    # Invert image
    eye =  cv2.bitwise_not(eye)

    return eye


def locate_eye(eye: np.ndarray, eye_points: np.ndarray) -> list:
    """ 
    Given a properly pre-processed image of an eye and its position, returns the x-y position
    of the eye in the frame.
  
    Parameters: 
    eye (np.ndarray): Pre-processed image of an eye.
    eye_points (np.ndarray): Points that shape the eye.
  
    Returns: 
    list: x, y and w, h of the iris.
  
    """
    contours, _ = cv2.findContours(eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    if contours == []: return None

    x, y, w, h = cv2.boundingRect(contours[0])
    
    ex, ey, _, _ = cv2.boundingRect(eye_points)

    return ex + x, ey + y, w, h


def main(frame: np.ndarray) -> dict:
    """ 
    Detects the eyes center position in 'frame'.

    If it does not find the eyes, returns None for each key
  
    Parameters: 
    frame (np.ndarray): uwu
  
    Returns: 
    dict: left and right eye centers.
  
    """
    left_eye_center, right_eye_center = None, None
    left_eye_points, right_eye_points = None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = detect_face(gray)

    if face is not None:
        shape = predictor(gray, face)
    
        left_eye_points, right_eye_points = get_eyes_keypoints(shape_to_np(shape))
        
        left_eye = crop_eye(gray, left_eye_points)
        right_eye = crop_eye(gray, right_eye_points)

        left_eye = preprocess_eye(left_eye)
        right_eye = preprocess_eye(right_eye)

        left_eye_coords = locate_eye(left_eye, left_eye_points)
        right_eye_coords = locate_eye(right_eye, right_eye_points)

        if (left_eye_coords and right_eye_coords):
            lx, ly, lw, lh = left_eye_coords
            rx, ry, rw, rh = right_eye_coords

            left_eye_center = np.array([lx + lw // 2, ly + lh // 2])
            right_eye_center = np.array([rx + rw // 2, ry + rh // 2])

    return {"eyes": {"left": left_eye_points, "right": right_eye_points}, "pupils": {"left": left_eye_center, "right": right_eye_center}}


def video(source, save_dir=False):
    cap = cv2.VideoCapture(source)

    if (not cap.isOpened()):  
        print("Error reading video file") 

    if (save_dir):
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4)) 
        
        size = (frame_width, frame_height)

        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(str(save_dir / "output.avi"),
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                fps, size)

    while (cap.isOpened()):
        ret, frame = cap.read()

        eyes = main(frame)["pupils"]

        if (eyes['left'] is not None and eyes['right'] is not None):
            cv2.circle(frame, tuple(eyes["left"].astype(np.int32)), 3, (0, 0, 255), -1)

            cv2.circle(frame, tuple(eyes["right"].astype(np.int32)), 3, (0, 0, 255), -1)

        if save_dir:
            writer.write(frame)
        else:        
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def images(imgs, save_dir=False):
    for img_path in imgs:
        img = cv2.imread(str(img_path))

        eyes = main(img)["pupils"]

        cv2.circle(img, tuple(eyes["left"].astype(np.int32)), 3, (0, 0, 255), -1)

        cv2.circle(img, tuple(eyes["right"].astype(np.int32)), 3, (0, 0, 255), -1)

        if save_dir:
            filename = save_dir / f"eyed-{img_path.name}"
            cv2.imwrite(str(filename), img)
        else:
            cv2.imshow('frame', img)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    from pathlib import Path

    path = Path.cwd() / "imgs"
    images(path.glob("*.jpg"), path)

    # video(0)
