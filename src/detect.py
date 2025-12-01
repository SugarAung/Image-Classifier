import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

model = tf.keras.models.load_model('sset_model.h5')
labels = ['Egg Tart', 'Salmon Sashimi', 'Unknown']

def predict_frame(image):
    image = ImageOps.fit(image, (75, 75), Image.LANCZOS).convert('RGB')
    image = np.asarray(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    confidence = np.max(prediction)
    label_index = np.argmax(prediction)
    return f"{labels[label_index]} ({confidence:.2f})" if confidence >= 0.7 else "Unknown (Low confidence)"

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite('temp.jpg', frame)
    image = Image.open('temp.jpg')
    label = predict_frame(image)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()