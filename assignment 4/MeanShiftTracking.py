import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    #check if there is an intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    #intersection area
    intersection_area = (x2 - x1) * (y2 - y1)
    
    #union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area
    
    return iou

def main():
    video_path = "KylianMbappe.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # code from tut 10
    ret, frame = cap.read()
    
    # detect a face on the first frame
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_boxes = face_detector.detectMultiScale(frame)
    #face_boxes = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(face_boxes) == 0:
        print('No face detected in the first frame')
        assert(False)
    
    # initialize the tracing window around the (first) detected face
    (x, y, w, h) = tuple(face_boxes[0])
    track_window = (x, y, w, h)
    
    # region of interest for tracking
    roi = frame[y:y+h, x:x+w]
    
    # convert the roi to HSV so we can construct a histogram of Hue 
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # why do we need this mask? (remember the cone?)
    # read the description for Figure 3 in the original Cam Shift paper: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.7673 
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    

    # form histogram of hue in the roi
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    
    # normalize the histogram array values so they are in the min=0 to max=255 range
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    # termination criteria for mean shift: 10 iteration or shift less than 1 pixel
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
    #store IoU values and frame numbers
    iou_values = []
    frame_numbers = []
    
    high_iou_frame = None
    low_iou_frame = None
    high_iou_boxes = None
    low_iou_boxes = None
    highest_iou = 0
    lowest_iou = 1.0
    frame_count = 1
    frames_with_high_iou = 0
    total_processed_frames = 0
    
    #starting from frame #2
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # code from tut 10
        # convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # histogram back projection using roi_hist 
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        
        # use meanshift to shift the tracking window
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        # display tracked window
        x_track, y_track, w_track, h_track = track_window
        tracked_box = (x_track, y_track, w_track, h_track)
        
        # detect face 
        detected_faces = face_detector.detectMultiScale(frame)
        #detected_faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(detected_faces) > 0:
            x_detect, y_detect, w_detect, h_detect = detected_faces[0]
            detected_box = (x_detect, y_detect, w_detect, h_detect)
            
            # calculate IoU between tracked and detected boxes
            iou = calculate_iou(tracked_box, detected_box)
            iou_values.append(iou)
            frame_numbers.append(frame_count)
            
            total_processed_frames += 1
            
            if iou > 0.5:
                frames_with_high_iou += 1
            
            #sstore if highest IoU (if over 50%)
            if iou > highest_iou and iou > 0.5:
                highest_iou = iou
                high_iou_frame = frame.copy()
                high_iou_boxes = (tracked_box, detected_box)
            
            #store if lowest IoU (if under 10%)
            if iou < lowest_iou and iou < 0.1:
                lowest_iou = iou
                low_iou_frame = frame.copy()
                low_iou_boxes = (tracked_box, detected_box)
            
            #tracked box in red
            cv2.rectangle(frame, (x_track, y_track), (x_track + w_track, y_track + h_track), (0, 0, 255), 2)
            cv2.putText(frame, "Tracked (Mean Shift)", (x_track, y_track - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            #detected box in green
            cv2.rectangle(frame, (x_detect, y_detect), (x_detect + w_detect, y_detect + h_detect), (0, 255, 0), 2)
            cv2.putText(frame, "Detected (Viola-Jones)", (x_detect, y_detect - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            #cv2.putText(frame, f"IoU: {iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            #cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display result
        cv2.imshow('Face Tracking', frame)
        
        if cv2.waitKey(33) & 0xFF == 27:  # wait a bit and exit is ESC is pressed
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    #percentage of frames with IoU > 50%
    if total_processed_frames > 0:
        high_iou_percentage = (frames_with_high_iou / total_processed_frames) * 100
    else:
        high_iou_percentage = 0
    
    #IoU over time
    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, iou_values)
    plt.xlabel('Frame Number')
    plt.ylabel('IoU')
    plt.title('Intersection over Union (IoU) over Time')
    plt.grid(True)
    plt.savefig('iou_over_time.png')
    plt.show()
    
    #save high IoU and low IoU
    if high_iou_frame is not None:
        tracked_box, detected_box = high_iou_boxes
        x_track, y_track, w_track, h_track = tracked_box
        x_detect, y_detect, w_detect, h_detect = detected_box
        
        cv2.rectangle(high_iou_frame, (x_track, y_track), (x_track + w_track, y_track + h_track), (0, 0, 255), 2)
        cv2.putText(high_iou_frame, "Tracked (Mean Shift)", (x_track, y_track - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(high_iou_frame, (x_detect, y_detect), (x_detect + w_detect, y_detect + h_detect), (0, 255, 0), 2)
        cv2.putText(high_iou_frame, "Detected (Viola-Jones)", (x_detect, y_detect - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(high_iou_frame, f"IoU: {highest_iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(high_iou_frame, f"Frame: {frame_numbers[iou_values.index(highest_iou)]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite('high_iou_frame.png', high_iou_frame)
    
    if low_iou_frame is not None:
        tracked_box, detected_box = low_iou_boxes
        x_track, y_track, w_track, h_track = tracked_box
        x_detect, y_detect, w_detect, h_detect = detected_box
        
        cv2.rectangle(low_iou_frame, (x_track, y_track), (x_track + w_track, y_track + h_track), (0, 0, 255), 2)
        cv2.putText(low_iou_frame, "Tracked (Mean Shift)", (x_track, y_track - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(low_iou_frame, (x_detect, y_detect), (x_detect + w_detect, y_detect + h_detect), (0, 255, 0), 2)
        cv2.putText(low_iou_frame, "Detected (Viola-Jones)", (x_detect, y_detect - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(low_iou_frame, f"IoU: {lowest_iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(low_iou_frame, f"Frame: {frame_numbers[iou_values.index(lowest_iou)]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite('low_iou_frame.png', low_iou_frame)
    
    #rresults
    print(f"Total frames processed: {total_processed_frames}")
    print(f"Frames with IoU > 50%: {frames_with_high_iou}")
    print(f"Percentage of frames with IoU > 50%: {high_iou_percentage:.2f}%")
    print(f"Highest IoU: {highest_iou:.2f}")
    print(f"Lowest IoU: {lowest_iou:.2f}")
    
    return {
        'total_frames': total_processed_frames,
        'high_iou_frames': frames_with_high_iou,
        'high_iou_percentage': high_iou_percentage,
        'highest_iou': highest_iou,
        'lowest_iou': lowest_iou
    }

if __name__ == "__main__":
    main()