from imports import *

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define objects to detect
object_descriptions = ["a person", "a dog", "a cat","a cap", "a car", "a tree", "a chair", "a football", "a cup", "a laptop", "a phone"]

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Selective Search Initialization
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply Selective Search for better object proposals
    ss.setBaseImage(frame)
    ss.switchToSelectiveSearchFast()  # Fast mode for real-time processing
    rects = ss.process()

    detected_objects = []
    max_proposals = min(len(rects), 20)  # Reduce number of regions to process

    for (x, y, w, h) in rects[:max_proposals]:
        if w < 50 or h < 50:  # Ignore very small boxes
            continue

        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224, 224))  # Resize to match CLIP input size

        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        text_inputs = torch.cat([clip.tokenize(desc) for desc in object_descriptions]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        similarity_scores = (image_features @ text_features.T).softmax(dim=-1)
        scores = similarity_scores.cpu().numpy()[0]

        max_score_index = np.argmax(scores)
        detected_object = object_descriptions[max_score_index]
        confidence = scores[max_score_index]

        if confidence > 0.85:  # Draw box only if confidence > 0.85
            detected_objects.append((x, y, w, h, detected_object, confidence))

    # Draw bounding boxes
    for (x, y, w, h, obj, conf) in detected_objects:
        label = f"{obj} ({conf:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Optimized Multi-Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
