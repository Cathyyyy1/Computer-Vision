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
        print('no face detected')
        assert(False)

    # initialize the tracing window around the (first) detected face
    (x, y, w, h) = tuple(face_boxes[0])
    track_window = (x, y, w, h)
    
    #  region of interest for tracking
    roi = frame[y:y+h, x:x+w]
    
    # convert the roi to HSV so we can construct a histogram of Hue
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    gradient_x = cv2.Sobel(blurred_roi, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_roi, cv2.CV_64F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
    mask = (magnitude > 0.88 * np.max(magnitude)).astype(np.uint8) * 255
    
    # form histogram of hue in the roi
    roi_hist = cv2.calcHist([angle.astype(np.float32)], [0], mask, [24], [0, 360])
    
    # normalize the histogram array values so they are in the min=0 to max=255 range
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    # termination criteria for mean shift: 10 iteration or shift less than 1 pixel
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
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
        # grab a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        frame_mask = (mag >  0.88 * np.max(mag)).astype(np.uint8) * 255
        
        # histogram back projection using roi_hist 
        back_proj = cv2.calcBackProject([ang.astype(np.float32)], [0], roi_hist, [0, 360], 1)
        back_proj = cv2.bitwise_and(back_proj, back_proj, mask=frame_mask)
        cv2.normalize(back_proj, back_proj, 0, 255, cv2.NORM_MINMAX)
        back_proj = back_proj.astype(np.uint8)
        
        # use meanshift to shift the tracking window
        ret, track_window = cv2.meanShift(back_proj, track_window, term_crit)
        
        # display tracked window
        x_track, y_track, w_track, h_track = track_window
        tracked_box = (x_track, y_track, w_track, h_track)
        
        # detect a face
        detected_faces = face_detector.detectMultiScale(frame)
        #detected_faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        #if faces are detected, calculate IoU
        if len(detected_faces) > 0:
            #Get the first detected face
            x_detect, y_detect, w_detect, h_detect = detected_faces[0]
            detected_box = (x_detect, y_detect, w_detect, h_detect)
            
            #calculate IoU between tracked and detected boxes
            iou = calculate_iou(tracked_box, detected_box)
            iou_values.append(iou)
            frame_numbers.append(frame_count)
            
            total_processed_frames += 1
            
            if iou > 0.5:
                frames_with_high_iou += 1
            
            if iou > highest_iou and iou > 0.8:
                highest_iou = iou
                high_iou_frame = frame.copy()
                high_iou_boxes = (tracked_box, detected_box)
            elif high_iou_frame is None and iou > 0.8:
                high_iou_frame = frame.copy()
                high_iou_boxes = (tracked_box, detected_box)
                highest_iou = iou
            
            if iou < lowest_iou and iou < 0.5:
                lowest_iou = iou
                low_iou_frame = frame.copy()
                low_iou_boxes = (tracked_box, detected_box)
            elif low_iou_frame is None and iou < 0.5:
                low_iou_frame = frame.copy()
                low_iou_boxes = (tracked_box, detected_box)
                lowest_iou = iou
            
            cv2.rectangle(frame, (x_track, y_track), (x_track + w_track, y_track + h_track), (0, 0, 255), 2)
            cv2.putText(frame, "Tracked (Gradient)", (x_track, y_track - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.rectangle(frame, (x_detect, y_detect), (x_detect + w_detect, y_detect + h_detect), (0, 255, 0), 2)
            cv2.putText(frame, "Detected (Viola-Jones)", (x_detect, y_detect - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(frame, f"IoU: {iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Face Tracking', frame)
        
        if cv2.waitKey(30) & 0xFF == 27:  # wait a bit and exit is ESC is pressed
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate percentage
    if total_processed_frames > 0:
        high_iou_percentage = (frames_with_high_iou / total_processed_frames) * 100
    else:
        high_iou_percentage = 0
    
    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, iou_values)
    plt.xlabel('Frame Number')
    plt.ylabel('IoU')
    plt.title('Intersection over Union (IoU) over Time (Gradient)')
    plt.grid(True)
    plt.savefig('gradient_iou_over_time.png')
    plt.show()
    
    if high_iou_frame is not None:
        tracked_box, detected_box = high_iou_boxes
        x_track, y_track, w_track, h_track = tracked_box
        x_detect, y_detect, w_detect, h_detect = detected_box
        
        cv2.rectangle(high_iou_frame, (x_track, y_track), (x_track + w_track, y_track + h_track), (0, 0, 255), 2)
        cv2.putText(high_iou_frame, "Tracked (Gradient)", (x_track, y_track - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(high_iou_frame, (x_detect, y_detect), (x_detect + w_detect, y_detect + h_detect), (0, 255, 0), 2)
        cv2.putText(high_iou_frame, "Detected (Viola-Jones)", (x_detect, y_detect - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(high_iou_frame, f"IoU: {highest_iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(high_iou_frame, f"Frame: {frame_numbers[iou_values.index(highest_iou)]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite('high_iou_gradient_frame.png', high_iou_frame)
    
    if low_iou_frame is not None:
        tracked_box, detected_box = low_iou_boxes
        x_track, y_track, w_track, h_track = tracked_box
        x_detect, y_detect, w_detect, h_detect = detected_box
        
        cv2.rectangle(low_iou_frame, (x_track, y_track), (x_track + w_track, y_track + h_track), (0, 0, 255), 2)
        cv2.putText(low_iou_frame, "Tracked (Gradient)", (x_track, y_track - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(low_iou_frame, (x_detect, y_detect), (x_detect + w_detect, y_detect + h_detect), (0, 255, 0), 2)
        cv2.putText(low_iou_frame, "Detected (Viola-Jones)", (x_detect, y_detect - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(low_iou_frame, f"IoU: {lowest_iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(low_iou_frame, f"Frame: {frame_numbers[iou_values.index(lowest_iou)]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite('low_iou_gradient_frame.png', low_iou_frame)
    
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