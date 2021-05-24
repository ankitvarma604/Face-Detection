import cv2
import os
import Predict as p

subjects = ["", "Ranbir", "Elvis Presley","Change it with your name"]


#load test images
test_img1 = cv2.imread("test-data/1.1.jfif")
test_img2 = cv2.imread("test-data/1.2.jfif")
test_img3 = cv2.imread("test-data/1.3.jpg")


#perform a prediction
predicted_img1 = p.predict(test_img1)
predicted_img2 = p.predict(test_img2)
predicted_img3 = p.predict(test_img3)
predicted_img3 = cv2.resize(predicted_img3, (700, 500))

print("Prediction complete")

#display images
cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
cv2.imshow(subjects[3], predicted_img3)

cv2.waitKey(0)
cv2.destroyAllWindows()