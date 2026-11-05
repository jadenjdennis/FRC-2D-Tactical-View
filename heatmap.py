import cv2
import numpy as np
import pandas as pd

field = cv2.imread("FieldMap.png")
h, w = field.shape[:2]


df = pd.read_csv(".csv").sort_values(by="timestamp").reset_index(drop=True)

timestamps = df["timestamp"].to_numpy()
xPos = df["x"].to_numpy()
yPos = df["y"].to_numpy()


#Decided to go w 20x20 for now, but lwk might wanna change this depending 
#how it turns out once Aarmaan finishes the other section
cellSize = 20
cols = int(np.ceil(w / cellSize))
rows = int(np.ceil(h / cellSize))
heatmap = np.zeros((rows, cols), dtype=np.float64)

maxTimeGap = 0.5


for i in range(len(timestamps) - 1):
    x1, y1 = xPos[i], yPos[i]
    x2, y2 = xPos[i+1], yPos[i+1]

    elapsedTime = timestamps[i+1] - timestamps[i]

    if ((elapsedTime <= 0) or (elapsedTime > maxTimeGap)):
        continue

    distance = np.hypot(x2 - x1, y2 - y1)
    numSteps = max(1, int(np.ceil(distance / (cellSize / 2))))

    for step in range(numSteps):
        fract = (step + 0.5) / numSteps
        curr_x = x1 + fract * (x2 - x1)
        curr_y = y1 + fract * (y2 - y1)

        if (not((0 <= curr_x < w) and (0 <= curr_y < h))):
            continue

        c = int(curr_x / cellSize)
        r = int(curr_y / cellSize)
        heatmap[r, c] += elapsedTime / numSteps


posValues = heatmap[heatmap > 0]
if (len(posValues) == 0):
    raise ValueError("No valid observations found.")

#Chose to use 99th percentile to avoid outliers bugging w the heatmap
max_val = np.percentile(posValues, 99)
norm = np.clip(heatmap / max_val * 255, 0, 255).astype(np.uint8)

resizedMap = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
coloredMap = cv2.applyColorMap(resizedMap, cv2.COLORMAP_JET)

#You gotta use addWeighted here cuz heatmap isn't transparent by default
alpha = 0.55
overlay = cv2.addWeighted(field, 1 - alpha, coloredMap, alpha, 0)


mask = resizedMap > 0

overlay[mask] = cv2.addWeighted(
    field[mask], 0.45, 
    coloredMap[mask], 0.55, 
    0
)

cv2.imwrite("FieldMap.png", overlay)
cv2.imshow("FieldMap", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()