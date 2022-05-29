#%%
import re, datetime,time, cv2, numpy as np, tensorflow as tf, sys

CHARS = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789" # exclude I, O
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
DECODE_DICT = {i:char for i, char in enumerate(CHARS)}

plates = set()

def capture_video(video_path):

    cap = cv2.VideoCapture(video_path)
    interpreter = tf.lite.Interpreter(model_path='detection.tflite')
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()

    recog_interpreter = tf.lite.Interpreter(model_path='LPRnet_finetune.tflite')
    recog_input_details = recog_interpreter.get_input_details()
    recog_output_details = recog_interpreter.get_output_details()
    recog_interpreter.resize_tensor_input(recog_input_details[0]['index'], (1, 24, 94, 3))
    recog_interpreter.allocate_tensors()

    frame_counter = 0
    print('starting')
    while cap.isOpened():
        frame_counter +=1 
        ret, frame = cap.read() # Capture each frame of video
        start = time.perf_counter()
        demo_frame = cv2.resize(frame, (680,480), interpolation=cv2.INTER_AREA)
        if frame_counter % 5 != 0:
        #    cv2.imshow('window-name', demo_frame)
            continue
        if not ret or frame is None:
            # raise LPRException("cap.read() returned invalid values!")
            break # Execution is finished
        resized = cv2.resize(frame, (320,320), interpolation=cv2.INTER_AREA)
        input_data = resized.astype(np.float32)          # Set as 3D RGB float array
        input_data /= 255.                               # Normalize
        input_data = np.expand_dims(input_data, axis=0)  # Batch dimension (wrap in 4D)

        # Initialize input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        boxes = interpreter.get_tensor(output_details[1]['index'])
        text = None
        # For index and confidence value of the first class [0]
        for i, confidence in enumerate(output_data[0]):
            if confidence > .3:
                text = execute_text_recognition_tflite(
                    boxes[0][i], frame,
                    recog_interpreter, recog_input_details, recog_output_details,
                )
                print(text,flush=True)
                plates.add(text)
                plates.add(text[:-1])

        cv2.imshow('window-name', demo_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break




def execute_text_recognition_tflite( boxes, frame, interpreter, input_details, output_details):
    x1, x2, y1, y2 = boxes[1], boxes[3], boxes[0], boxes[2]
    save_frame = frame[
        max( 0, int(y1*1079) ) : min( 1079, int(y2*1079) ),
        max( 0, int(x1*1920) ) : min( 1920, int(x2*1920) )
    ]

    # Execute text recognition

    test_image = cv2.resize(save_frame,(94,24))/256
    test_image = np.expand_dims(test_image,axis=0)
    test_image = test_image.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    decoded = tf.keras.backend.ctc_decode(output_data,(24,),greedy=False)
    text = ""
    for i in np.array(decoded[0][0][0]):
        if i >-1:
            text += DECODE_DICT[i]
    # Do nothing if text is empty
    if not len(text): return 
    license_plate = text
    text[:3].replace("0",'O')

    return text
#%%
if __name__ == '__main__':
    args = sys.argv
    capture_video(args[1])
    print(plates)
# %%
capture_video("C:\\Users\\carlos\\Pictures\\accuracy_test_vid.mp4")
print(plates)
#%%
cv2.destroyAllWindows()
ann_plates = [
"NBD9011"
,"NB01379"
,"DAL6890"
,"NAK5393"
,"NGA7363"
,"CAS4417"
,"GAD5174"
,"MAN9630"
,"ABF7189"
,"NBM8676"
,"DBZ9983"
,"AAP2120"
,"NAT5828"
,"NBA8392"
,"AAI4126"
,"DAT1598"
,"NBN1363"
,"CAV1176"
,"NCL7306"
,"NCZ4359"
,"NDL1305"
,"ABG2566"
,"NAN9630"
,"CAV5486"
,"NAU8156"
,"AIA7073"
,"CAB9519"
,"NAI1177"
,"NDA6696"
,"ABE8672"
,"IAB1365"
,"NBE8960"
,"NCS6929"
,"NCQ2857"
,"NFW9734"
,"ARA9085"
,"ABH3746"
,"SHN1218"
,"NCR6808"
,"AVA6890"
,"NCB3178"
,"NCL6522"
,"IAB6499"
,"ABX5587"
,"NDD2441"
,"NXS861"
,"NEH4544"
,"AAR3155" 
,"CAF5424"
,"AKA4369"
,"NDA4522"
,"NCG8369"
,"AQA7707"
,"AKA6692"
,"NDI7665"
,"NCZ3277"
,"NEF2759"
,"NFU2938"
,"NBX2829"
,"AAJ3751"
]
ctr = 0
for p in ann_plates:
    if p in plates:
        print(p)
        ctr+=1 
print(ctr,len(ann_plates))
# %% 10/60
