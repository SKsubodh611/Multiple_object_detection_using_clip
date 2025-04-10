from imports import *
import time

# Load the CLIP model with optimizations
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # Set to evaluation mode

# Use half precision if on CUDA
if device == "cuda":
    model.half()

# Define objects with simpler descriptions (fewer tokens)
object_descriptions = ["person", "dog", "cat", "car", "tree", "chair", "football", "cup"]
text_inputs = torch.cat([clip.tokenize(desc) for desc in object_descriptions]).to(device)

# Precompute text features once (they don't change)
with torch.no_grad():
    if device == "cuda":
        text_inputs = text_inputs.half()
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Configuration
FRAME_SKIP = 3  # Process every 3rd frame
MIN_REGION_SIZE = 300  # Minimum area for region consideration
CONFIDENCE_THRESHOLD = 0.8
MAX_REGIONS = 20  # Limit number of regions to process

def get_region_proposals(image):
    # More efficient than contour detection
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    regions = ss.process()
    
    # Filter by size and limit quantity
    filtered = []
    for x, y, w, h in regions:
        if w * h > MIN_REGION_SIZE:
            filtered.append((x, y, w, h))
            if len(filtered) >= MAX_REGIONS:
                break
    return filtered

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # Get region proposals
    regions = get_region_proposals(frame)
    if not regions:
        continue

    # Batch processing of regions
    batch_images = []
    valid_regions = []
    
    for (x, y, w, h) in regions:
        roi = frame[y:y+h, x:x+w]
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            continue
        
        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image)
        batch_images.append(image_input)
        valid_regions.append((x, y, w, h))

    if not batch_images:
        continue

    # Process all regions in one batch
    image_inputs = torch.stack(batch_images).to(device)
    if device == "cuda":
        image_inputs = image_inputs.half()

    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Matrix multiplication for all regions at once
        similarity_scores = (image_features @ text_features.T).softmax(dim=-1)
        scores = similarity_scores.cpu().numpy()

    # Process results
    detected_objects = []
    for i, (x, y, w, h) in enumerate(valid_regions):
        max_score_index = np.argmax(scores[i])
        confidence = scores[i][max_score_index]
        
        if confidence > CONFIDENCE_THRESHOLD:
            detected_objects.append((
                x, y, w, h, 
                object_descriptions[max_score_index], 
                confidence
            ))

    # Draw results
    for (x, y, w, h, obj, conf) in detected_objects:
        label = f"{obj} ({conf:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Optimized Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()