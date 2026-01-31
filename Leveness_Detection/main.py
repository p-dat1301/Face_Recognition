from headPoseEstimation import HeadPoseActimation
import cv2

success, image = HeadPoseActimation()

if success:
    print("Ok!")
    cv2.imshow("Image Output", image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
else:
    print("Not Ok!")
