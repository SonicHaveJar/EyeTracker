FRONTAL_FACE_URL=https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
FACE_LANDMARK=http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
FACE_LANDMARK_FILE=shape_predictor_68_face_landmarks.dat.bz2

wget $FRONTAL_FACE_URL

wget -N $FACE_LANDMARK -O $FACE_LANDMARK_FILE
bzip2 -dk $FACE_LANDMARK_FILE
rm $FACE_LANDMARK_FILE
