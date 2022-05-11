import re, datetime, cv2, numpy as np, tensorflow as tf, sys, multiprocessing as mp,time
from charset_normalizer import detect
from tokenize import detect_encoding

CHARS = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789" # exclude I, O
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
DECODE_DICT = {i:char for i, char in enumerate(CHARS)}
framecounter = 0
detectcounter = 0
processes = []

interpreter = tf.lite.Interpreter(model_path='detection.tflite')
interpreter.allocate_tensors()

recog_interpreter = tf.lite.Interpreter(model_path='recognition.tflite')
recog_interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

recog_input_details = recog_interpreter.get_input_details()
recog_output_details = recog_interpreter.get_output_details()


def capture_video(video_path,frame_buffer,results):
    global detectcounter
    global framecounter
    t1 = time.perf_counter()
    time1 =0 
    time2 = 0
    cap = cv2.VideoCapture(video_path)
    print('starting capture')
    while cap.isOpened():
        framecounter +=1 
        ret, frame = cap.read() # Capture each frame of video
        if not ret or frame is None:
            # raise LPRException("cap.read() returned invalid values!")
            break # Execution is finished
        
        if not results.empty():
            res = results.get(False)
            detectcounter += 1
            print(res)
        frame_buffer.put(frame)
        framecounter += 1
        if framecounter >= 15:
            framecounter = 0
            t2 = time.perf_counter()
            print(15/(t2-t1))
            print(detectcounter/(t2-t1))
            detectcounter = 0
            t1 = t2
            
            
def execute_detection(boxes_queue,frame_queue,frame_buffer):
    print('starting detection')
    while True:
        if frame_buffer.empty():
            continue
        frame = frame_buffer.get()
        resized = cv2.resize(frame, (320,320), interpolation=cv2.INTER_AREA)

        input_data = resized.astype(np.float32)          # Set as 3D RGB float array
        input_data /= 255.                               # Normalize
        input_data = np.expand_dims(input_data, axis=0)  # Batch dimension (wrap in 4D)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Confidence values
        # output_data = [ n = # of classes [ confidence values of boxes ] ]

        # details = [
        #     {"index": [ n = # of classes [ confidence values of boxes ] ]},
        #     {"index": [ n = # of classes [ details of boxes ] ]}
        # ]
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Bounding boxes
        boxes = interpreter.get_tensor(output_details[1]['index'])
        for i, confidence in enumerate(output_data[0]):
            if confidence > .3:
                frame_queue.put(frame)
                boxes_queue.put(boxes[0][i])



def execute_text_recognition_tflite( boxes_queue, frame_queue, results):
    print('starting recognition')
    while True:
        if frame_queue.empty():
            continue
        boxes = boxes_queue.get()
        frame = frame_queue.get()
        x1, x2, y1, y2 = boxes[1], boxes[3], boxes[0], boxes[2]
        save_frame = frame[
            max( 0, int(y1*1079) ) : min( 1079, int(y2*1079) ),
            max( 0, int(x1*1920) ) : min( 1920, int(x2*1920) )
        ]

        # Execute text recognition

        test_image = cv2.resize(save_frame,(94,24))/256
        test_image = np.expand_dims(test_image,axis=0)
        test_image = test_image.astype(np.float32)
        recog_interpreter.set_tensor(recog_input_details[0]['index'], test_image)
        recog_interpreter.invoke()
        output_data = recog_interpreter.get_tensor(recog_output_details[0]['index'])
        decoded = tf.keras.backend.ctc_decode(output_data,(24,),greedy=False)
        text = ""
        for i in np.array(decoded[0][0][0]):
            if i >-1:
                text += DECODE_DICT[i]
        license_plate = text
        text[:3].replace("0",'O')
        results.put(text)

if __name__ == '__main__':

    frame_buffer = mp.Queue(10)
    frame_queue = mp.Queue(10)
    boxes_queue = mp.Queue(10)
    results = mp.Queue()

    # Start streaming
    p = mp.Process(target=capture_video,
                    args=(sys.argv[1],frame_buffer,results),
                    daemon=True)
    p.start()
    processes.append(p)

    # Activation of detection model
    p = mp.Process(target=execute_detection,
                    args=(boxes_queue,frame_queue,frame_buffer),
                    daemon=True)
    p.start()
    processes.append(p)

    # Activation of recognition model
    p = mp.Process(target=execute_text_recognition_tflite,
                    args=(boxes_queue,frame_queue, results),
                    daemon=True)
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
