import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_paths(file1: str, file2: str) -> tuple[str, str]:
    with open(file1, 'r', encoding='utf-8') as f1:
        path1 = f1.read().strip()

    with open(file2, 'r', encoding='utf-8') as f2:
        path2 = f2.read().strip()

    return path1, path2


current_dir = os.path.dirname(os.path.abspath(__file__))
pics_dir = os.path.join(current_dir, "SatellitePics")
select_pic_txt = "src/main/resources/analyze/firstImagePath.txt"
select_pic_txt2 = "src/main/resources/analyze/secondImagePath.txt"
path1, path2 = load_paths(select_pic_txt, select_pic_txt2)
select_pic = path1
select_pic2 = path2
img1 = cv2.imread(os.path.join(pics_dir, select_pic))
img2 = cv2.imread(os.path.join(pics_dir, select_pic2))

# debugging
# print(f"{pics_dir + "\\"}2020-02-07-00_00_2020-02-07-23_59_Sentinel-2_L2A_True_color.jpg")
# print(img1)
# print(f"{pics_dir + "\\"}2025-03-07-00_00_2025-03-07-23_59_Sentinel-2_L2A_True_color.jpg")
# print(img2)

hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

water_color_threshold1 = np.array([1, 1, 1])
upper_water_color_threshold1 = np.array([190, 255, 255])

water_color_threshold2 = np.array([1, 1, 1])
upper_water_color_threshold2 = np.array([190, 255, 255])

mask_img1 = cv2.inRange(hsv_img1, water_color_threshold1, upper_water_color_threshold1)
mask_img2 = cv2.inRange(hsv_img2, water_color_threshold2, upper_water_color_threshold2)

area_img1 = np.sum(mask_img1 > 0)
area_img2 = np.sum(mask_img2 > 0)

percent_change = ((area_img2 - area_img1) / area_img1) * 100


def detect_and_adjust_lake(img, hsv_img, lower_threshold, upper_threshold, target_area=372000, max_iterations=24,
                           tolerance=5000):
    """
    Args:
        img: Original BGR image
        hsv_img: HSV converted image
        lower_threshold: Starting lower threshold for water detection [h, s, v]
        upper_threshold: Upper threshold for water detection [h, s, v]
        target_area: Target area in pixels
        max_iterations: Maximum adjustment iterations
        tolerance: Acceptable difference from target area

    Returns:
        adjusted_threshold: Final lower threshold values
        lake_mask: Mask of the detected lake
        lake_area: Area of the detected lake in pixels
        vis_img: Visualization image with contour drawn
    """
    current_threshold = lower_threshold.copy()

    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    best_mask = None
    best_contour = None
    best_area = 0
    best_threshold = current_threshold.copy()
    smallest_area_diff = float('inf')

    for i in range(max_iterations):
        water_mask = cv2.inRange(hsv_img, current_threshold, upper_threshold)

        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel, iterations=4)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=5)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        main_contour = None
        max_area = 0
        min_dist = float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:
                continue

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

                score = area - (dist * 10)

                if score > max_area:
                    max_area = score
                    main_contour = contour
                    min_dist = dist

        if main_contour is not None:
            lake_mask = np.zeros_like(water_mask)
            cv2.drawContours(lake_mask, [main_contour], 0, 255, -1)

            lake_area = cv2.contourArea(main_contour)

            area_diff = abs(lake_area - target_area)
            if area_diff < smallest_area_diff:
                smallest_area_diff = area_diff
                best_mask = lake_mask.copy()
                best_contour = main_contour.copy()
                best_area = lake_area
                best_threshold = current_threshold.copy()

            if area_diff <= tolerance:
                break

            area_difference = lake_area - target_area

            adjustment_h = int(area_difference / 60000) + (1 if area_difference > 0 else -1)
            adjustment_s = int(area_difference / 40000) + (2 if area_difference > 0 else -2)

            current_threshold[0] += adjustment_h
            current_threshold[1] += adjustment_s

            current_threshold[0] = max(0, min(179, current_threshold[0]))
            current_threshold[1] = max(0, min(255, current_threshold[1]))
        else:
            current_threshold[0] = max(0, current_threshold[0] - 5)
            current_threshold[1] = max(0, current_threshold[1] - 10)

    if best_contour is not None:
        vis_img = img.copy()
        cv2.drawContours(vis_img, [best_contour], 0, (10, 40, 255), 3)
        return best_threshold, best_mask, best_area, vis_img
    else:
        water_mask = cv2.inRange(hsv_img, lower_threshold, upper_threshold)
        area = np.sum(water_mask > 0)
        return lower_threshold, water_mask, area, img.copy()


adjusted_threshold1, lake_mask1, lake_area1, vis_img1 = detect_and_adjust_lake(
    img1, hsv_img1, water_color_threshold1, upper_water_color_threshold1
)

adjusted_threshold2, lake_mask2, lake_area2, vis_img2 = detect_and_adjust_lake(
    img2, hsv_img2, water_color_threshold2, upper_water_color_threshold2
)

# percent_change_corrected = ((corrected_px2 - corrected_px) / corrected_px) * 100

# debugging
print(f"{area_img1} px")
print(f"{area_img2} px")
# if corrected_color is not None:
#    print(f"{corrected_color} 1st image color correction")
# if corrected_color2 is not None:
#    print(f"{corrected_color2} 2nd image color correction")
# if corrected_px is not None:
#    print(f"{corrected_px} px after 1st image correction")
# if corrected_px2 is not None:
#    print(f"{corrected_px2} px after 2nd image correction")
print(f"{percent_change}%")


# print(f"{percent_change_corrected} after correction")

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


print(f"Lake area 1: {lake_area1} px")
print(f"Lake area 2: {lake_area2} px")
print(f"Adjusted threshold 1: {adjusted_threshold1}")
print(f"Adjusted threshold 2: {adjusted_threshold2}")
print(f"Percent change: {percent_change}%")

plt.figure(figsize=(15, 10))

# Show original with contours
plt.subplot(2, 2, 1)
plt.title(f"Jezioro Tałty, {select_pic[8:10]} {monthConverter(select_pic[5:7]).capitalize()} {select_pic[:4]}")
plt.imshow(cv2.cvtColor(vis_img1, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title(f"Jezioro Tałty, {select_pic2[8:10]} {monthConverter(select_pic2[5:7]).capitalize()} {select_pic2[:4]}")
plt.imshow(cv2.cvtColor(vis_img2, cv2.COLOR_BGR2RGB))

# Show masks
plt.subplot(2, 2, 3)
plt.title(f"Powierzchnia: ~{lake_area1} px²")
plt.imshow(lake_mask1, cmap='Blues')

plt.subplot(2, 2, 4)
plt.title(f"Powierzchnia: ~{lake_area2} px²")
plt.imshow(lake_mask2, cmap='Blues')

plt.tight_layout()
plt.savefig(f"{current_dir}/{"analyzedChart"}.png", transparent=True)
"""
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title(f"Jezioro Tałty, {select_pic[8:10]} {monthConverter(select_pic[5:7]).capitalize()} {select_pic[:4]}\n Powierzchnia: ~{px_area()} px²")
plt.imshow(mask_img1, cmap='Blues')
plt.subplot(1,2,2)
plt.title(f"Jezioro Tałty, {select_pic2[8:10]} {monthConverter(select_pic2[5:7]).capitalize()} {select_pic2[:4]}\n Powierzchnia: ~{px_area2()} px²")
plt.imshow(mask_img2, cmap='Blues')
"""
# plt.show()
