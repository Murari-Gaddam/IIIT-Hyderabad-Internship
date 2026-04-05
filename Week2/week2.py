from ultralytics import YOLO

# Load the latest YOLO26n model (nano version for speed)
model = YOLO("yolo26n.pt")

# Run inference on an image from a URL
results = model("Week2/birds.jpg")

# Display the results with bounding boxes
results[0].show()
