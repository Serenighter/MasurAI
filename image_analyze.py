import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
pics_dir = os.path.join(current_dir, "Satellite pics")
select_pic = input("Please enter the first image file's name: ")
select_pic2 = input("Please enter the second image file's name: ")
img_2020 = cv2.imread(f"{pics_dir + "\\" + select_pic}")
img_2025 = cv2.imread(f"{pics_dir + "\\" + select_pic2}")

#debugging
#print(f"{pics_dir + "\\"}2020-02-07-00_00_2020-02-07-23_59_Sentinel-2_L2A_True_color.jpg")
#print(img_2020)
#print(f"{pics_dir + "\\"}2025-03-07-00_00_2025-03-07-23_59_Sentinel-2_L2A_True_color.jpg")
#print(img_2025)

hsv_img2020 = cv2.cvtColor(img_2020, cv2.COLOR_BGR2HSV)
hsv_img2025 = cv2.cvtColor(img_2025, cv2.COLOR_BGR2HSV)

water_color = np.array([60, 30, 30]) #15, 30, 20
upperwater_color = np.array([140, 255, 255]) #130, 255, 255

mask_img2020 = cv2.inRange(hsv_img2020, water_color, upperwater_color)
mask_img2025 = cv2.inRange(hsv_img2025, water_color, upperwater_color)

area_img2020 = np.sum(mask_img2020 > 0)
area_img2025 = np.sum(mask_img2025 > 0)

percent_change = ((area_img2025 - area_img2020) / area_img2020) * 100

if (percent_change > 100 or area_img2025 > 110000 or area_img2020 > 110000) :
    water_color[0] += 12
    water_color[1] += 31
    print(water_color)
    mask_img2020 = cv2.inRange(hsv_img2020, water_color, upperwater_color)
    mask_img2025 = cv2.inRange(hsv_img2025, water_color, upperwater_color)

print(f"{area_img2020} px")
print(f"{area_img2025} px")
print(f"{percent_change}%")
print(type(percent_change), type(area_img2020))

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title("Tałty w 2020")
plt.imshow(mask_img2020, cmap='Blues')
plt.subplot(1,2,2)
plt.title("Tałty w 2025")
plt.imshow(mask_img2025, cmap='Blues')
plt.show()