import detect_lincense
import split_horizontally
import matplotlib.pyplot as plt
import cv2


image = "./images/car.jpg"
plate = detect_lincense.detect_lincense(image)
plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
plt.show()
split_figures = split_horizontally.split_lincense_horizontally(plate)
for i in split_figures:
    plt.imshow(i, cmap="gray")
    plt.show()
