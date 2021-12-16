import numpy as np
import cv2
import dlib
import math
import pickle
import pandas as pd
import winsound


# Initialization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('facial-landmarks/shape_predictor_68_face_landmarks.dat')
loaded_model = pickle.load(open('LogisticRegression.sav','rb'))
sound_path = 'mixkit-emergency-alert-alarm-1007.wav'

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 400)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

number_of_frame_to_average = 90

count_number_of_frame_to_average = 0

feature_list_to_normalise = []



test_element = [np.nan,np.inf,-np.inf]
# support definition
def face_recognition(frame):
    rects = detector(frame,0)
    if len(rects) == 0:
        return frame, np.zeros((68,2),dtype=int)
    for i in rects:
        x_start = i.left()
        y_start = i.top()
        x_end = i.right()
        y_end = i.bottom()
        shape = predictor(frame, i)
        coords = np.zeros((shape.num_parts, 2), dtype=int)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        for (x, y) in coords:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    return frame, coords

def distance(a,b):
    return ( (a[0]-b[0]) **2 + (a[1]-b[1]) **2 ) ** 0.5

def eye_aspect_ratio(shape):
    return ( ( distance(shape[37],shape[41]) + distance(shape[38],shape[40]) ) / ( 2 * distance(shape[36],shape[39]) ) )

def mouth_aspect_ratio(shape):
    return ( distance(shape[51],shape[57]) / distance(shape[48],shape[54]) )

def pupil_circularity(shape):
    Area = ( (distance(shape[37],shape[40]) / 2 ) **2 ) * math.pi
    perimeter = distance(shape[36],shape[37]) + distance(shape[37],shape[38]) + distance(shape[38],shape[39]) \
                + distance(shape[39],shape[40]) + distance(shape[40],shape[41]) + distance(shape[36],shape[41])
    return ( 4 * math.pi * Area ) / ( perimeter ** 2 )

def mouth_aspect_ratio_over_eye_aspect_ratio(shape):
    return mouth_aspect_ratio(shape) / eye_aspect_ratio(shape)

def mean_std(feature_list):
    feature_list = np.array(feature_list)
    EAR_mean = np.mean(feature_list[:,0])
    MAR_mean = np.mean(feature_list[:,1])
    PUC_mean = np.mean(feature_list[:,2])
    MOE_mean = np.mean(feature_list[:,3])
    EAR_std = np.std(feature_list[:,0])
    MAR_std = np.std(feature_list[:,1])
    PUC_std = np.std(feature_list[:,2])
    MOE_std = np.std(feature_list[:,3])
    mean_std = [EAR_mean,MAR_mean,PUC_mean,MOE_mean,EAR_std,MAR_std,PUC_std,MOE_std]
    if 0 in mean_std:
        return [0,0,0,0,0,0,0,0]
    return mean_std

def pre_process_before_normalise(array_points):
    EAR = eye_aspect_ratio(array_points)
    MAR = mouth_aspect_ratio(array_points)
    PUC = pupil_circularity(array_points)
    MOE = mouth_aspect_ratio_over_eye_aspect_ratio(array_points)
    feature = [EAR,MAR,PUC,MOE]
    return feature

def pre_process_to_feature(array_points,mean_std_to_normalise):
    feature = pre_process_before_normalise(array_points)
    EAR_N = ( feature[0] - mean_std_to_normalise[0] ) / mean_std_to_normalise[4]
    MAR_N = ( feature[1] - mean_std_to_normalise[1] ) / mean_std_to_normalise[5]
    PUC_N = ( feature[2] - mean_std_to_normalise[2] ) / mean_std_to_normalise[6]
    MOE_N = ( feature[3] - mean_std_to_normalise[3] ) / mean_std_to_normalise[7]
    feature.append(EAR_N)
    feature.append(MAR_N)
    feature.append(PUC_N)
    feature.append(MOE_N)
    feature = np.array(feature)
    if np.isin(feature,test_element).any():
        return pd.DataFrame()
    feature = feature.reshape((1,8))
    result_df = pd.DataFrame(columns=['EAR','MAR','PUC','MOE','EAR_N','MAR_N','PUC_N','MOE_N'], data=feature)
    #result_df = clean_dataset(result_df)
    return result_df

def mean_std_normalise():
    capvid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capvid.set(cv2.CAP_PROP_FPS, 30)
    feature_list_to_normalise = []

    while len(feature_list_to_normalise)<100:
        ret, frame=capvid.read()
        (frame,shape) = face_recognition(frame)
        cv2.putText(frame, 'Calibrate...', bottomLeftCornerOfText,
                    font, fontScale, fontColor, lineType)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        if 0 in shape:
            # print('No face detected')
            continue

        feature_list_to_normalise.append(pre_process_before_normalise(shape))

    mean_std_to_normalise = mean_std(feature_list_to_normalise)
    if mean_std_to_normalise == [0,0,0,0,0,0,0,0]:
        return mean_std_normalise()
    capvid.release()
    cv2.destroyAllWindows()
    return mean_std_to_normalise

def run_model():
    return 0


mean_std_to_normalise = mean_std_normalise()
#Main
capvid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capvid.set(cv2.CAP_PROP_FPS,30)
while (True):
    # Capture frame-by-frame
    ret, frame = capvid.read()

    # Our operations on the frame come here
    #frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)
    (frame,shape) = face_recognition(frame)
    if 0 in shape:
        #print('No face detected')
        cv2.putText(frame, 'No face detected' , bottomLeftCornerOfText,
                    font, fontScale, fontColor, lineType)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    # Calibrate to normalise

    # Run model
    result_array = []
    while count_number_of_frame_to_average < 90:
        feature = pre_process_to_feature(shape,mean_std_to_normalise)
        if len(feature.index) == 0:
            continue
        y_test = loaded_model.predict(feature)
        result_array.append(y_test)
        count_number_of_frame_to_average += 1

    if count_number_of_frame_to_average == 90:
        y_test_final = sum(result_array) / 90
        if y_test_final >= 0.5:
            cv2.putText(frame, 'Normal', bottomLeftCornerOfText,
                        font, fontScale, fontColor, lineType)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            winsound.PlaySound(sound_path, winsound.SND_ASYNC)
            cv2.putText(frame, 'Drownsiness', bottomLeftCornerOfText,
                        font, fontScale, fontColor, lineType)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        count_number_of_frame_to_average = 0
        result_array = []





# When everything done, release the capture
capvid.release()
cv2.destroyAllWindows()

