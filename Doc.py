import cv2
import numpy as np

width_img = 640
heigth_img = 480

# cap = cv2.VideoCapture("http://192.168.1.71:4747/video")
# --- get video stream from ip camera
cap = cv2.VideoCapture("http://192.168.1.71:4747/video")
# --- set width and height for video
cap.set(3, width_img)  # ---- setting for height
cap.set(4, heigth_img)  # ---- setting for width
cap.set(10, 160)  # ---- setting for brightness


# --- some process on each frame of vedeo
def preProcessing(img):
    # --- make gray scale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # --- insert a Gaussian blur for removing noise
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0.5)
    # --- find edge of each things
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    # --- dilation and erodsion in frame
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres, imgCanny


# --- tring to find boggest countour`s area and return it
def getCountours(img):
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxArea = 0
    biggest = np.array([])
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            # cv2.drawContours(img_countour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    # --- draw the biggest countour
    cv2.drawContours(img_countour, biggest, -1, (255, 0, 0), 20)
    return biggest


# --- this function sort each of 4 x,y of each corner of doc for best wrapping
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew
    pass


def getWrap(img, biggest):
    imgOutput = imgCropped = np.ones((width_img, heigth_img))
    if biggest != []:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [width_img, 0], [0, heigth_img], [width_img, heigth_img]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (width_img, heigth_img))
        imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
        imgCropped = cv2.resize(imgCropped, (width_img, heigth_img))
    return imgCropped


while True:
    bool_result, img = cap.read()
    img = cv2.resize(img, (width_img, heigth_img))
    img_countour = img
    imgThres, h = preProcessing(img)
    biggest = getCountours(imgThres)
    print(biggest)
    wrap = getWrap(img, biggest)
    cv2.imshow("image", img_countour)
    cv2.imshow("wrap", wrap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
