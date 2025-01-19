from flask import Flask,render_template,url_for,Response
import numpy as np
import tensorflow as tf
app = Flask(__name__)
import cv2
model = tf.keras.models.load_model('face_mask_detector.h5')
camera = cv2.VideoCapture(0)


def draw_label(img,text,pos,bg_color):

    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
    end_x  = pos[0] + text_size[0][0]+2
    end_y  = pos[1] + text_size[0][1] -2
    cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)
def detect_face_mask(img):
    y_pred = model.predict(img.reshape(1,224,224,3))
    if y_pred[0][0] >=0.5:
        return 1
    else:
        return 0


def generate_frames():
    while True:
        success,frame = camera.read()
        if not success:
            break
        else:
            detector = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
            img = cv2.resize(frame,(224,224))
            coods = detector.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            for x,y,w,h in coods:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness = 3)
            y_pred = detect_face_mask(img)
            if y_pred == 0:
                draw_label(frame,"Mask",(30,30),(0,255,0))
            else:
                draw_label(frame,"No Mask",(30,30),(0,0,255))
            ret,buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')
if __name__ == '__main__':
    app.run(debug = True)

    
    
