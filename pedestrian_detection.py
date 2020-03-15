import cv2
import numpy as np
import sys
from random import seed
from random import randint

# Helper for median function. Checks if current color is unique, if so add to channel else ignore
def check_if_unique(color, current_color_channel):
    if color not in current_color_channel:
        current_color_channel.append(color)

# Updates background model. If current change is too large, stick with previous background model for that block.
def update_background_model(previous, current, threshold, ksize):
    bg_model = np.zeros_like(current, dtype=np.uint8)
    height, width = current.shape[:2]
    thresh_diff = ksize * ksize * 0.3
    for i in range(0, height, ksize):
        for j in range(0, width, ksize):
            bound_x = i + ksize
            bound_y = j + ksize
            changed = 0
            if i + ksize > height:
                bound_x = height - i
            if j + ksize > width:
                bound_y = width - j
            for x in range(i, bound_x):
                for y in range(j, bound_y):
                    prev = previous[x][y].astype(np.float32)
                    temp = current[x][y].astype(np.float32)
                    difference = abs(prev - temp)
                    if difference > threshold:
                        bg_model[i][j] = previous[i][j]
                        changed += 1
            if changed > thresh_diff:
                for q in range(i, bound_x):
                    for r in range(j, bound_y):
                        bg_model[q][r] = previous[q][r]
            else:
                for q in range(i, bound_x, 1):
                    for r in range(j, bound_y, 1):
                        bg_model[q][r] = current[q][r]
    return bg_model

# Applies median filter to each pixel in background model over time instead of spatially.
def median_filter_over_time(first_twenty_frames):
    length = len(first_twenty_frames)
    height = first_twenty_frames[0].shape[0]
    width = first_twenty_frames[0].shape[1]
    background_model = np.zeros_like(first_twenty_frames[0], dtype=np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            r = []
            g = []
            b = []
            for current_frame in range(0, length):
                # Append the current pixel at i,j if it is not already in array
                check_if_unique(first_twenty_frames[current_frame][i][j][0], r)
                check_if_unique(first_twenty_frames[current_frame][i][j][1], g)
                check_if_unique(first_twenty_frames[current_frame][i][j][2], b)
            # Sort each channel array
            r = np.sort(r)
            g = np.sort(g)
            b = np.sort(b)
            # Assign each channel of the background model to the median of the channel
            background_model[i][j][0] = r[int(len(r)/2)]
            background_model[i][j][1] = g[int(len(g)/2)]
            background_model[i][j][2] = b[int(len(b)/2)]
    return background_model

# Helper for difference function. Checks a block of pixels to see if a large enough percentage of the block has changed outside the threshold. If so, set mask value at this block to black. Otherwise, set it to white.
def threshold_set_to_black(mask, gray, thresh, i, j, bound_x, bound_y):
    adj = 0
    over_thresh = 0
    count = 0
    for x in range(i, bound_x):
        for y in range(j, bound_y):
            if mask[x][y] == 0:
                over_thresh += 1
            elif (y - bound_y >= 0 and x - bound_x >= 0) and (mask[x][y - bound_y] == 0 or mask[x - bound_x][y] == 0):
                if gray[x][y] > thresh:
                    adj += 1
            count += 1
    if count != 0:
        if over_thresh / count > 0.45 or adj / count > 0.2:
            val = 0
        else:
            val = 255
        for x in range(i, bound_x):
            for y in range(j, bound_y):
                mask[x][y] = val
    return mask

# Helper function to check to see if a block is surrounded by blocks that have been set to black. If so, set to black. Otherwise, if no blacked out blocks are adjacent to it, then it is likely noise, so set to white.
def check_if_surrounded_or_alone(mask, gray, thresh, i, j, bound_x, bound_y, height, width):
    surrounded = 0
    count = 0
    for x in range(i, bound_x):
        if j - 1 > 0:
            if mask[x][j-1] == 0:
                surrounded += 1
        if j + bound_y < width:
            if mask[x][j+bound_y] == 0:
                surrounded += 1
        count += 2
    for y in range(j, bound_y):
        if i - 1 > 0:
            if mask[i-1][y] == 0:
                surrounded += 1 
        if i + bound_y < height:
            if mask[i+bound_y][y] == 0:
                surrounded += 1
        count += 2
    if count != 0:
        if surrounded / count >= 0.75:
            val = 0
        elif surrounded == 0:
            val = 255
        else:
            return mask
        for x in range(i, bound_x):
            for y in range(j, bound_y):
                mask[x][y] = val
    return mask

# Difference function for robust block comparison.
def diff(bg_model, gray, thresh, kernel, robust):
    height, width = gray.shape[:2]
    thresh_diff = kernel * kernel * 0.3
    mask = np.full_like(gray, 255, dtype=np.uint8)
    ksize = kernel
    for i in range(0, height):
        for j in range(0, width):
            bg = bg_model[i][j].astype(np.float32)
            g = gray[i][j].astype(np.float32)
            difference = abs(bg - g)
            if difference > thresh:
                mask[i][j] = 0
    ksize = kernel
    for i in range(0, height, ksize):
        for j in range(0, width, ksize):
            adj = 0
            over_thresh = 0
            count = 0
            bound_x = i + ksize
            bound_y = j + ksize
            if bound_x >= height:
                bound_x = height - i
            if bound_y >= width:
                bound_y = width - j
            mask = threshold_set_to_black(mask, gray, thresh, i, j, bound_x, bound_y)
            mask = check_if_surrounded_or_alone(mask, gray, thresh, i, j, bound_x, bound_y, height, width)
        if j < width:
            bound_x = height - i
            bound_y = width - j
            mask = threshold_set_to_black(mask, gray, thresh, i, j, bound_x, bound_y)
            mask = check_if_surrounded_or_alone(mask, gray, thresh, i, j, bound_x, bound_y, height, width)
    return mask

# Difference function for pixel-by-pixel comparison.
def diff_precise(bg_model, gray, thresh):
    height, width = gray.shape[:2]
    mask = np.full_like(gray, 255, dtype=np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            bg = bg_model[i][j].astype(np.float32)
            g = gray[i][j].astype(np.float32)
            difference = abs(bg - g)
            if difference > thresh:
                mask[i][j] = 0
    return mask

# Compare against background and flag as changed.
def flag_as_changed(frame, bg_model, kernel, threshold):
    height, width = frame.shape[:2]
    frame = cv2.medianBlur(frame, 9)
    average = 0
    counter = 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i in range(0, height, kernel):
        for j in range(0, width, kernel):
            changed = []
            if i + kernel >= width:
                kernel = width-i
            if j + kernel >= height:
                kernel = height-j
            for k in range(i, i+kernel):
                for l in range(j, j+kernel):
                    bg_mod_pixel = bg_model[k][l].astype(np.float32)
                    current_gray_pixel = gray[k][l].astype(np.float32)
                    if abs(bg_mod_pixel - current_gray_pixel) > threshold:
                        changed.append([i+k,j+l])
            length_of_changed_pixels = len(changed)
            kernel_size = kernel*kernel
            if length_of_changed_pixels > 0:
                percent_changed = length_of_changed_pixels/kernel_size
                if percent_changed >= 0.5:
                    for y in range(i, i+kernel):
                        for x in range(j, j+kernel):
                            gray[y][x] = 0
                else:
                    for y in range(i, i+kernel):
                        for x in range(j, j+kernel):
                            gray[y][x] = bg_model[y][x]
    return gray

# Writes edited footage out to a file
def edit_footage(path_to_vid, robust):

    # Read footage into a video capture object
    cap = cv2.VideoCapture(path_to_vid)

    # Check if video loaded successfully
    frames_to_read = cap.isOpened()

    # If something went wrong, exit function
    if not frames_to_read:
        print("Unable to read camera feed")
        return 0

    # Default resolutions of the frame and adjusted to half initial size: convert from float to integer
    frame_width = int(cap.get(3) * 0.5)
    frame_height = int(cap.get(4) * 0.5)

    # Define the codec and create VideoWriter object. The output is stored in the file at the '.avi' path
    out = cv2.VideoWriter('/Users/ktsutter/Downloads/car_edited.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_height, frame_width))

    # Variables to hold number of frames for background model, threshold for change, kernel size, and the array to hold the frames for the background model.
    num_frames_for_model = 20
    num_frames_read_in = 0
    threshold = 10
    kernel = 10
    frames_for_bg_model = []

    # Read in the first twenty frames for the initial background model.
    for i in range(0, num_frames_for_model):
        read_correctly, frame = cap.read()
        if read_correctly:
            frame = cv2.resize(frame, (frame_width, frame_height), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            frames_for_bg_model.append(frame)
            num_frames_read_in += 1
        else:
            break
    # Generate background model, then apply median blur filter and convert to grayscale.
    bg_model = median_filter_over_time(frames_for_bg_model)
    bg_model = cv2.medianBlur(bg_model, 9)
    bg_model = cv2.cvtColor(bg_model, cv2.COLOR_BGR2GRAY)

    # As long as there are still frames to read, continue.
    while frames_to_read:
        if read_correctly:
            current_frames = []
            # Read in frames for background model.
            for i in range(0, num_frames_for_model):
                read_correctly, current_frame = cap.read()
                if read_correctly:
                    # Resize the current frame.
                    current_frame = cv2.resize(current_frame, (frame_width, frame_height), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                    current_frames.append(current_frame)
                    num_frames_read_in += 1
                else:
                    break
            length = len(current_frames)
            # If at least one frame was read in, edit it.
            if length > 0:
                for i in range(0, length):
                    if robust:
                        gray = flag_as_changed(current_frames[i], bg_model, kernel, threshold)
                        bg_mask = diff(bg_model, gray, threshold, kernel, robust)
                    else:
                        gray = cv2.cvtColor(current_frames[i], cv2.COLOR_BGR2GRAY)
                        bg_mask = diff_precise(bg_model, gray, threshold)
                    # Copy the resulting mask to the current frame
                    result = cv2.bitwise_and(current_frames[i], current_frames[i], mask=bg_mask)
                    result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
                    # Save frame to file
                    out.write(result)
                potential_bg_model = median_filter_over_time(current_frames)
                potential_bg_model = cv2.cvtColor(potential_bg_model, cv2.COLOR_BGR2GRAY)
                # If user has selected robust for block/vehicle detection, update the background model in blocks.
                if robust:
                    bg_model = update_background_model(bg_model, potential_bg_model, threshold*2, kernel)
                # Otherwise, set the background model to the result from the current 20 frames.
                else:
                    bg_model = potential_bg_model

    # Release video capture and write objects and close all frames
    cap.release()
    out.release()

# Main function. Accepts two command line arguments: path to video to be edited, and keyword 'robust' or 'precise,' depending on which version the user wants.
def __main():
    video = sys.argv[1]
    preferred_result = sys.argv[2]
    robust = True
    if preferred_result == "precise":
        robust = False
    edit_footage(video, robust)

__main()
