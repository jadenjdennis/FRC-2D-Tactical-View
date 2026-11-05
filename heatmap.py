import cv2
import numpy as np
import pandas as pd

field = cv2.imread("FieldMap.png")

h, w = field.shape[:2]

print("Width:", w)
print("Height:", h)

df = pd.read_csv("data.csv")

df = pd.sort_values(by="timestamp")

