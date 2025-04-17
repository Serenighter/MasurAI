import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
pics_dir = os.path.join(current_dir, "Satellite pics")
select_pic = input("Please enter the first image file's name: ")
select_pic2 = input("Please enter the second image file's name: ")
img1 = cv2.imread(os.path.join(pics_dir, select_pic))
img2 = cv2.imread(os.path.join(pics_dir, select_pic2))

#debugging
#print(f"{pics_dir + "\\"}2020-02-07-00_00_2020-02-07-23_59_Sentinel-2_L2A_True_color.jpg")
#print(img1)
#print(f"{pics_dir + "\\"}2025-03-07-00_00_2025-03-07-23_59_Sentinel-2_L2A_True_color.jpg")
#print(img2)

hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

water_color_threshold1 = np.array([60, 30, 30])
upper_water_color_threshold1 = np.array([140, 255, 255])

water_color_threshold2 = np.array([60, 30, 30])
upper_water_color_threshold2 = np.array([140, 255, 255])

mask_img1 = cv2.inRange(hsv_img1, water_color_threshold1, upper_water_color_threshold1)
mask_img2 = cv2.inRange(hsv_img2, water_color_threshold2, upper_water_color_threshold2)

area_img1 = np.sum(mask_img1 > 0)
area_img2 = np.sum(mask_img2 > 0)

percent_change = ((area_img2 - area_img1) / area_img1) * 100

def adjust_thresholds_for_images(hsv_img1, hsv_img2, threshold1, threshold2, upper_threshold1, upper_threshold2, target_area=520000, max_iterations=24, tolerance=15000):
    current_threshold1 = threshold1.copy()
    for i in range(max_iterations):
        mask1 = cv2.inRange(hsv_img1, current_threshold1, upper_threshold1)
        current_area1 = np.sum(mask1 > 0)
        
        if abs(current_area1 - target_area) <= tolerance:
            break
        
        area_difference = current_area1 - target_area
        adjustment_h = int(area_difference / 50000) + (1 if area_difference > 0 else -1)
        adjustment_s = int(area_difference / 30000) + (2 if area_difference > 0 else -2)
        
        current_threshold1[0] += adjustment_h
        current_threshold1[1] += adjustment_s
        
        current_threshold1[0] = max(0, min(179, current_threshold1[0]))
        current_threshold1[1] = max(0, min(255, current_threshold1[1]))
    
    mask1 = cv2.inRange(hsv_img1, current_threshold1, upper_threshold1)
    area1 = np.sum(mask1 > 0)
    
    current_threshold2 = threshold2.copy()
    for i in range(max_iterations):
        mask2 = cv2.inRange(hsv_img2, current_threshold2, upper_threshold2)
        current_area2 = np.sum(mask2 > 0)
        
        if abs(current_area2 - target_area) <= tolerance:
            break
        
        area_difference = current_area2 - target_area
        adjustment_h = int(area_difference / 50000) + (1 if area_difference > 0 else -1)
        adjustment_s = int(area_difference / 30000) + (2 if area_difference > 0 else -2)
        
        current_threshold2[0] += adjustment_h
        current_threshold2[1] += adjustment_s
        
        current_threshold2[0] = max(0, min(179, current_threshold2[0]))
        current_threshold2[1] = max(0, min(255, current_threshold2[1]))
    
    mask2 = cv2.inRange(hsv_img2, current_threshold2, upper_threshold2)
    area2 = np.sum(mask2 > 0)
    
    return current_threshold1, area1, mask1, current_threshold2, area2, mask2

corrected_color, corrected_px, mask_img1, corrected_color2, corrected_px2, mask_img2 = adjust_thresholds_for_images(hsv_img1, hsv_img2, water_color_threshold1, water_color_threshold2, upper_water_color_threshold1, upper_water_color_threshold2)

percent_change_corrected = ((corrected_px2 - corrected_px) / corrected_px) * 100

#debugging
print(f"{area_img1} px")
print(f"{area_img2} px")
if corrected_color is not None:
    print(f"{corrected_color} 1st image color correction")
if corrected_color2 is not None:
    print(f"{corrected_color2} 2nd image color correction")
if corrected_px is not None:
    print(f"{corrected_px} px after 1st image correction")
if corrected_px2 is not None:
    print(f"{corrected_px2} px after 2nd image correction")
print(f"{percent_change}%")
print(f"{percent_change_corrected} after correction")

def monthConverter(string):
    try:
        stringInteger = int(string)
    except ValueError:
        return "Invalid Month"
    
    match stringInteger:
        case 1:
            return "styczeń"
        case 2:
            return "luty"
        case 3:
            return "marzec"
        case 4:
            return "kwiecień"
        case 5:
            return "maj"
        case 6:
            return "czerwiec"
        case 7:
            return "lipiec"
        case 8:
            return "sierpień"
        case 9:
            return "wrzesień"
        case 10:
            return "październik"
        case 11:
            return "listopad"
        case 12:
            return "grudzień"
        case _:
            return "Invalid Month"

def px_area():
    if corrected_px is not None:
        return str(corrected_px)
    else:
        return str(area_img1)

def px_area2():
    if corrected_px2 is not None:
        return str(corrected_px2)
    else:
        return str(area_img2)

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title(f"Jezioro Tałty, {select_pic[8:10]} {monthConverter(select_pic[5:7]).capitalize()} {select_pic[:4]}\n Powierzchnia: ~{px_area()} px²")
plt.imshow(mask_img1, cmap='Blues')
plt.subplot(1,2,2)
plt.title(f"Jezioro Tałty, {select_pic2[8:10]} {monthConverter(select_pic2[5:7]).capitalize()} {select_pic2[:4]}\n Powierzchnia: ~{px_area2()} px²")
plt.imshow(mask_img2, cmap='Blues')
#plt.savefig(f"{current_dir}/{select_pic[0:10]}-{select_pic2[0:10]}.png", transparent=True)
plt.show()
