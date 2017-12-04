import detect_lincense
import split_horizontally
import matplotlib.pyplot as plt
import cv2


image = "./car.jpg"
plate = detect_lincense.detect_lincense(image)
split_figures = split_horizontally.split_lincense_horizontally(plate)
for i in split_figures:
    plt.imshow(i, cmap="gray")
    plt.show()
