import cv2 as cv
import mediapipe as mp

handStatus = {
    "RIGHT_THUMB": False, "RIGHT_INDEX": False, "RIGHT_MIDDLE": False, "RIGHT_RING": False, "RIGHT_PINKY": False, "RIGHT_THUMB": False, 
    "LEFT_THUMB": False, "LEFT_INDEX": False, "LEFT_MIDDLE": False, "LEFT_RING": False, "LEFT_PINKY": False, "LEFT_THUMB": False, }
fingerList = {"INDEX": 6, "MIDDLE": 10, "RING": 14, "PINKY": 18}

url = None 
cap = cv.VideoCapture(url) #either use the URL variable for DriodCam or simply use 0 as a parameter
cap.set(3, 300)
cap.set(4, 500) 
cap.set(5, 24)

mpUtils = mp.solutions.drawing_utils
mpHand = mp.solutions.hands

with mpHand.Hands(static_image_mode = False,max_num_hands = 2,min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as hand:      
    while cap.isOpened():
        _, frame = cap.read()
        if not _:
            break
        
        #Main Code   
        frame = cv.flip(frame, 1) 
        h, w = frame.shape[0], frame.shape[1]   
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hand.process(rgb)
        # temp = []
        count = 0
        
        # Reset Dictionary
        for k in handStatus:
            handStatus[k] = False
            
        # Extracting Data 
        if result.multi_handedness and result.multi_hand_landmarks:
            for handLm, handLmId in zip(result.multi_hand_landmarks, result.multi_handedness):
                # mpUtils.draw_landmarks(frame, handLm, mpHand.HAND_CONNECTIONS)
                lm = handLm.landmark
                lmInfo = handLmId.classification[0].label
                
                # Finger Logic
                for index, x in enumerate(lm):
                    lm[index].x = int(lm[index].x * w)
                    lm[index].y = int(lm[index].y * h)
                    
                for key, value in fingerList.items():
                    if lm[value+2].y < lm[value].y:
                        handStatus[lmInfo.upper()+"_"+key] = True
                        
                # Thumb logic
                if lmInfo.upper() == 'LEFT':
                    if lm[4].x > lm[3].x:
                        handStatus[lmInfo.upper()+"_THUMB"] = True
                else:
                    if lm[4].x < lm[3].x:
                        handStatus[lmInfo.upper()+"_THUMB"] = True
                        
        # What To Draw
        for key, value in handStatus.items():
            if value:
                # temp.append(key)
                count+= 1 
        
        # Drawing and Exit
        cv.putText(frame, f"the total number of fingers on the screen{count}", (0, 30), cv.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)
        cv.imshow("Finger Counter", frame) 
        if cv.waitKey(10) & 0xFF==ord('q'):
            break

cap.release()
cv.destroyAllWindows()
        
    