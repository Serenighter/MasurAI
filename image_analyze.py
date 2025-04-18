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

def detect_main_lake(img, hsv_img, lower_threshold, upper_threshold):
    """
    Detect the main lake in the center of the image using watershed segmentation
    
    Args:
        img: Original BGR image
        hsv_img: HSV converted image
        lower_threshold: Lower threshold for water detection [h, s, v]
        upper_threshold: Upper threshold for water detection [h, s, v]
        
    Returns:
        lake_mask: Binary mask of the main lake
        contours: Contours of the lake
        area: Area of the lake in pixels
    """
    # Initial water detection
    water_mask = cv2.inRange(hsv_img, lower_threshold, upper_threshold)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find all contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return empty results
    if not contours:
        return water_mask, None, np.sum(water_mask > 0)
    
    # Get image center coordinates
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Find the contour closest to the center of the image
    main_contour = None
    min_distance = float('inf')
    
    for contour in contours:
        # Only consider contours with significant area
        area = cv2.contourArea(contour)
        if area < 1000:  # Adjust this threshold as needed
            continue
            
        # Calculate center of contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate distance to image center
            distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
            # Update if this contour is closer to center and has significant area
            if distance < min_distance:
                min_distance = distance
                main_contour = contour
    
    # If no main contour found, return the whole water mask
    if main_contour is None:
        return water_mask, contours, np.sum(water_mask > 0)
    
    # Create mask with just the main lake
    lake_mask = np.zeros_like(water_mask)
    cv2.drawContours(lake_mask, [main_contour], 0, 255, -1)
    
    # Calculate area
    area = cv2.contourArea(main_contour)
    
    return lake_mask, main_contour, area

def adjust_lake_thresholds(hsv_img1, hsv_img2, img1, img2, threshold1, threshold2, 
                          upper_threshold1, upper_threshold2, target_area=520000, 
                          max_iterations=20, tolerance=10000):
    """
    Detect the main lake and adjust thresholds for accurate detection
    """
    # Initial lake detection
    lake_mask1, lake_contour1, initial_area1 = detect_main_lake(
        img1, hsv_img1, threshold1, upper_threshold1
    )
    
    # Initial lake detection for second image
    lake_mask2, lake_contour2, initial_area2 = detect_main_lake(
        img2, hsv_img2, threshold2, upper_threshold2
    )
    
    # Adjust first image threshold
    current_threshold1 = threshold1.copy()
    for i in range(max_iterations):
        # Apply threshold and detect lake
        mask_img1 = cv2.inRange(hsv_img1, current_threshold1, upper_threshold1)
        lake_mask1, _, current_area1 = detect_main_lake(
            img1, hsv_img1, current_threshold1, upper_threshold1
        )
        
        if abs(current_area1 - target_area) <= tolerance:
            break
        
        area_difference = current_area1 - target_area
        adjustment_h = int(area_difference / 50000) + (1 if area_difference > 0 else -1)
        adjustment_s = int(area_difference / 30000) + (2 if area_difference > 0 else -2)
        
        current_threshold1[0] += adjustment_h
        current_threshold1[1] += adjustment_s
        
        current_threshold1[0] = max(0, min(179, current_threshold1[0]))
        current_threshold1[1] = max(0, min(255, current_threshold1[1]))
    
    # Adjust second image threshold
    current_threshold2 = threshold2.copy()
    for i in range(max_iterations):
        # Apply threshold and detect lake
        mask_img2 = cv2.inRange(hsv_img2, current_threshold2, upper_threshold2)
        lake_mask2, _, current_area2 = detect_main_lake(
            img2, hsv_img2, current_threshold2, upper_threshold2
        )
        
        if abs(current_area2 - target_area) <= tolerance:
            break
        
        area_difference = current_area2 - target_area
        adjustment_h = int(area_difference / 50000) + (1 if area_difference > 0 else -1)
        adjustment_s = int(area_difference / 30000) + (2 if area_difference > 0 else -2)
        
        current_threshold2[0] += adjustment_h
        current_threshold2[1] += adjustment_s
        
        current_threshold2[0] = max(0, min(179, current_threshold2[0]))
        current_threshold2[1] = max(0, min(255, current_threshold2[1]))
    
    # Final detection with adjusted thresholds
    final_lake_mask1, final_contour1, area1 = detect_main_lake(
        img1, hsv_img1, current_threshold1, upper_threshold1
    )
    
    final_lake_mask2, final_contour2, area2 = detect_main_lake(
        img2, hsv_img2, current_threshold2, upper_threshold2
    )
    
    # Create visualization with contours
    vis_img1 = img1.copy()
    vis_img2 = img2.copy()
    
    if final_contour1 is not None:
        cv2.drawContours(vis_img1, [final_contour1], 0, (0, 255, 0), 2)
    
    if final_contour2 is not None:
        cv2.drawContours(vis_img2, [final_contour2], 0, (0, 255, 0), 2)
    
    return (current_threshold1, final_lake_mask1, area1, current_threshold2, 
            final_lake_mask2, area2, vis_img1, vis_img2)

results = adjust_lake_thresholds(
    hsv_img1, hsv_img2, img1, img2,
    water_color_threshold1, water_color_threshold2,
    upper_water_color_threshold1, upper_water_color_threshold2
)

corrected_color, lake_mask1, corrected_px, corrected_color2, lake_mask2, corrected_px2, vis_img1, vis_img2 = results

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

plt.figure(figsize=(15, 10))

#original with contours
plt.subplot(2, 2, 1)
plt.title(f"Jezioro Tałty, {select_pic[8:10]} {monthConverter(select_pic[5:7]).capitalize()} {select_pic[:4]}")
plt.imshow(cv2.cvtColor(vis_img1, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title(f"Jezioro Tałty, {select_pic2[8:10]} {monthConverter(select_pic2[5:7]).capitalize()} {select_pic2[:4]}")
plt.imshow(cv2.cvtColor(vis_img2, cv2.COLOR_BGR2RGB))

#masks
plt.subplot(2, 2, 3)
plt.title(f"Powierzchnia: ~{corrected_px} px²")
plt.imshow(lake_mask1, cmap='Blues')

plt.subplot(2, 2, 4)
plt.title(f"Powierzchnia: ~{corrected_px2} px²")
plt.imshow(lake_mask2, cmap='Blues')

plt.tight_layout()
plt.show()
"""
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title(f"Jezioro Tałty, {select_pic[8:10]} {monthConverter(select_pic[5:7]).capitalize()} {select_pic[:4]}\n Powierzchnia: ~{px_area()} px²")
plt.imshow(mask_img1, cmap='Blues')
plt.subplot(1,2,2)
plt.title(f"Jezioro Tałty, {select_pic2[8:10]} {monthConverter(select_pic2[5:7]).capitalize()} {select_pic2[:4]}\n Powierzchnia: ~{px_area2()} px²")
plt.imshow(mask_img2, cmap='Blues')
"""
#plt.savefig(f"{current_dir}/{select_pic[0:10]}-{select_pic2[0:10]}.png", transparent=True)
#plt.show()
