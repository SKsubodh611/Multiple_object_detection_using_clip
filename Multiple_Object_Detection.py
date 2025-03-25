from imports import * 

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define the objects you want to detect
object_descriptions = ["a person", "a dog", "a cat", "a car", "a tree", "a chair", "a football", "a cup"]

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to perform selective search
def get_region_proposals(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()  # Faster but fewer boxes
    rects = ss.process()
    return rects[:50]  # Limit to top 50 region proposals

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break


frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:  # Skip every 2 frames
        continue

    # Convert to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get region proposals
    # regions = get_region_proposals(frame)


    regions = cv2.findContours(cv2.Canny(frame, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    regions = [cv2.boundingRect(cnt) for cnt in regions if cv2.contourArea(cnt) > 500]

    detected_objects = []

    for (x, y, w, h) in regions:
        roi = frame[y:y+h, x:x+w]
        if roi.shape[0] < 20 or roi.shape[1] < 20:  # Skip very small regions
            continue
        

    # roi = cv2.resize(roi, (128, 128))  # Resize to 128x128 for faster processing

        # Convert region to PIL Image and preprocess
        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        # Prepare text inputs
        text_inputs = torch.cat([clip.tokenize(desc) for desc in object_descriptions]).to(device)

        # Compute similarity scores
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        
        similarity_scores = (image_features @ text_features.T).softmax(dim=-1)
        scores = similarity_scores.cpu().numpy()[0]

        # Get the most likely object for this region
        max_score_index = np.argmax(scores)
        detected_object = object_descriptions[max_score_index]
        confidence = scores[max_score_index]

        # Filter out low confidence detections
        if confidence > 0.8:
            detected_objects.append((x, y, w, h, detected_object, confidence))

    # Draw bounding boxes on detected objects
    for (x, y, w, h, obj, conf) in detected_objects:
        label = f"{obj} ({conf:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Multi-Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
