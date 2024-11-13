import mediapipe as mp
import cv2
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

mapping = {
    99: 'CLEAR'
}

for i in range(1, 99):
    mapping[i] = str(i)

changeThreshold = 5
itr = 0
currentGesture = None
tempGesture = None

enteredGesturesStr = ''
xCoord = 180
inc = 10
totalDPM = 0
totalErrRate = 0

def transform(x, y, a, org):
    a = 2 * math.pi - a
    t = np.array([[1, 0, -x],
            [0, 1, -y],
            [0, 0, 1]])
    s = np.array([[1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]])
    r = np.array([[np.cos(a), np.sin(a), 0],
            [-np.sin(a), np.cos(a), 0],
            [0, 0, 1]])
    return np.matmul(np.matmul(np.matmul(np.array(org), t.T), s.T), r.T)

def transform_points(keypoints):
    x0 = keypoints[0][0]
    y0 = keypoints[0][1]
    x9 = keypoints[9][0]
    y9 = keypoints[9][1]
    a = math.degrees(math.atan2(y9 - y0, x9 - x0))
    a = (((a + 360) % 360) + 90) % 360
    a = math.radians(a)
    keypoints_transformed = transform(x0, y0, a, keypoints)
    return keypoints_transformed

def getFingerCounts(handLandmarks, label):
    count = 0
    sum = 0
    msg = ''
    multiplier = 10 if label == 'left' else 1
    msg += f'{label}: '

    dist_thumb_tip = handLandmarks[4][0]
    dist_thumb_base = handLandmarks[3][0]
    checker = dist_thumb_tip * dist_thumb_base
    if checker < 0:
        pass
    elif abs(dist_thumb_tip) > abs(dist_thumb_base):
        count += 1
        sum += (5 * multiplier)
        msg += f'T{int(dist_thumb_tip)}, {int(dist_thumb_base)} '
    dist_index_tip = abs(handLandmarks[8][1])
    dist_index_base = abs(handLandmarks[6][1])
    if dist_index_tip > dist_index_base:
        count += 1
        sum += (1 * multiplier)
        msg += f'I{int(dist_index_tip)}, {int(dist_index_base)}'
    dist_middle_tip = abs(handLandmarks[12][1])
    dist_middle_base = abs(handLandmarks[10][1])
    if dist_middle_tip > dist_middle_base:
        count += 1
        sum += (1 * multiplier)
        msg += f'M{int(dist_middle_tip)}, {int(dist_middle_base)} '
    dist_ring_tip = abs(handLandmarks[16][1])
    dist_ring_base = abs(handLandmarks[14][1])
    if dist_ring_tip > dist_ring_base:
        count += 1
        sum += (1 * multiplier)
        msg += f'R{int(dist_ring_tip)}, {int(dist_ring_base)} '
    dist_pinky_tip = abs(handLandmarks[20][1])
    dist_pinky_base = abs(handLandmarks[18][1])
    if dist_pinky_tip > dist_pinky_base:
        count += 1
        sum += (1 * multiplier)
        msg += f'P{int(dist_pinky_tip)}, {int(dist_pinky_base)} '
    return count, sum, msg


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=2) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        isLeft = False
        isRight = False
        
        left_hand_points = []
        right_hand_points = []

        cv2.putText(image, f'Abacus Gestures:', (5, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(image, f'{enteredGesturesStr}_', (180, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 153), 1)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                if (results.multi_handedness[num].classification[0].label == 'Left'):
                    isLeft = True
                    for i in range(21):
                        left_hand_points.append([int(hand.landmark[i].x * frameWidth), int(hand.landmark[i].y * frameHeight), 1])
                    left_key_points_transformed = transform_points(left_hand_points)

                if (results.multi_handedness[num].classification[0].label == 'Right'):
                    isRight = True
                    for i in range(21):
                        right_hand_points.append([int(hand.landmark[i].x * frameWidth), int(hand.landmark[i].y * frameHeight), 1])
                    right_key_points_transformed = transform_points(right_hand_points)

            leftSum, rightSum = 0, 0
            leftMsg, rightMsg = '', ''
            if isLeft:
                leftCount, leftSum, leftMsg = getFingerCounts(left_key_points_transformed, 'left')
                
            if isRight:
                rightCount, rightSum, rightMsg = getFingerCounts(right_key_points_transformed, 'right')


            sum = leftSum + rightSum
            try:
                letter = mapping[sum]
                if letter != tempGesture:
                    tempGesture = letter
                    itr = 0
                else:
                    if currentGesture != tempGesture:
                        itr += 1
                        if itr >= changeThreshold:
                            itr = 0
                            currentGesture = tempGesture
                if currentGesture != None:
                    cv2.putText(image, currentGesture, (xCoord, 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), 1)

            except Exception as e:
                if sum == 0:
                    itr = 0
                    if currentGesture != None:
                        if currentGesture == 'CLEAR':
                            enteredGesturesStr = ''
                            xCoord = 180
                            itr = 0
                        
                        else:
                            enteredGesturesStr += currentGesture + ' '
                            xCoord += inc * 2
                        
                        currentGesture = None
                        tempGesture = None
                else:
                    if tempGesture != None:
                        itr += 1
                        if itr >= changeThreshold:
                            tempGesture = None
                            currentGesture = None
                            itr = 0

        cv2.imshow('Test hand', image)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
cap.release()