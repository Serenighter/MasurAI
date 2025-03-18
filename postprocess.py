import cv2
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
pics_dir = os.path.join(current_dir, "Satellite pics")
results_dir = os.path.join(current_dir, "results")
os.makedirs(pics_dir, exist_ok=True)
print(f"directory: {pics_dir}")

for img_name in os.listdir(pics_dir):
    if img_name.endswith(".jpg"):
        img_path = os.path.join(pics_dir, img_name)
        img = cv2.imread(img_path)

        denoised = cv2.GaussianBlur(img, (5, 5), 0)

        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        output_path = os.path.join(results_dir, img_name)
        cv2.imwrite(output_path, enhanced)

print("Success!")
