import numpy as np
import cv2



capvid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while(True):
    # Capture frame-by-frame
    ret, frame = capvid.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
capvid.release()
cv2.destroyAllWindows()
