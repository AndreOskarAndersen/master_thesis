import cv2

def show_frame(frame):
    """
    Function for visualizing a frame.
    """
    
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1)