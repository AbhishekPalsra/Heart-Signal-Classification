import numpy as np
import pywt
import cv2
import time
print(np.__version__)
from PIL import Image
import tflite_runtime.interpreter as tflite
interpreter=tflite.Interpreter(model_path='/home/pi/tensorflow-lite-custom-model-main/examples/lite/examples/object_detection/raspberry_pi/efficientdet_lite0.tflite')
interpreter.allocate_tensors()
input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()
print(input_details)
print(output_details)


def ecg_to_image(signal, voices_per_octave=16):
    siglen = len(signal)
    scales = np.geomspace(1, 200, num=voices_per_octave * 8)
    coefficients, _ = pywt.cwt(signal, scales, 'cmor2.5-1.0', method="fft")
    cfs = np.abs(coefficients)
    return cfs

# Step 3: Convert CWT coefficients to RGB image
def apply_colormap(cfs):
    cfs_normalized = (cfs - cfs.min()) / (cfs.max() - cfs.min()) * 255
    cfs_uint8 = cfs_normalized.astype(np.uint8)
    cfs_resized = cv2.resize(cfs_uint8, (224, 224))
    image = cv2.applyColorMap(cfs_resized, cv2.COLORMAP_JET)
    return image

# Step 4: Load TFLite model and classify image
def load_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    # Normalize
    image=image.astype(np.float32)
    image = np.expand_dims(image, axis=0)     # Add batch dimension


    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Get prediction scores (logits or probabilities)
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Determine class and confidence
    predicted_class = int(np.argmax(output))
    confidence = float(np.max(output))

    return predicted_class, confidence

##input
print(pywt.__version__)
import time
start=time.time()
signal=np.load('/home/pi/tensorflow-lite-custom-model-main/examples/lite/examples/object_detection/raspberry_pi/test_2.npy')
print(signal)
cfs = ecg_to_image(signal,16)
scalogram_image = apply_colormap(cfs)
# Predict class
predicted_class, confidence = predict_image(interpreter, scalogram_image)

print("Predicted Class:", predicted_class)
print("Confidence Score:", confidence)

print("Execution Time: {:.2f} seconds".format(time.time() - start))
    
    # Predict class

end=time.time()
timet=start-end
print(f"Time Taken:{timet:.4f} seconds")
 # Optional: Show imag
cv2.imshow("Scalogram", scalogram_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





    
