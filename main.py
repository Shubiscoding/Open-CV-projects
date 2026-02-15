import cv2 as cv 
import mediapipe as mp

imgList = [cv.imread(x) for x in [r"images\miku.png", r"images\monkey1.png", r"images\monkey2.png"]]
reactStatus = {"miku":False, "monkey1":False, "monkey2": False}

url = "http://192.168.29.65:4747/video"
cap = cv.VideoCapture(url)
cap.set(3, 300), 
cap.set(4, 300),
cap.set(5, 24)

mpUtils = mp.solutions.drawing_utils
mpHolistic = mp.solutions.holistic

with mpHolistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        _, frame = cap.read()
        if not _:
            break
    
    # Reseting
        for k in reactStatus:
            reactStatus[k] = False
        
        frame = cv.flip(frame, 1)
        h, w = frame.shape[:2]
        
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = holistic.process(rgb)
        
        if result.pose_landmarks:
            lm = result.poseLm.landmark
            lmPos = []    
            for index, x in enumerate(lm):
                lmPos.append([int(lm[index].x * w), int(lm[index].y * h) ])
                
            if (lmPos[16][1] < lmPos[12][1]) and (lmPos[15][1] < lmPos[11][1]):
                reactStatus["miku"] = True

        for x, y in reactStatus.items():
            if y:
                cv.putText(frame, f"{x}", (0, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)   
                break
            else:
                cv.putText(frame, "None", (0, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)   
                    
        cv.imshow("Frame", frame)
        cv.waitKey(0)   
cv.destroyAllWindows()