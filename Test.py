import cv2, dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model

IMG_SIZE = (32, 32)

total = 0
left = 0
right = 0
both = 0
text = ""
text2 = ""
eye_l = True
eye_r = True
eye_b = True
check = True
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('models/2022_06_04_15_14_07.h5')
model.summary()
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

def checkBlink(pred_l, pred_r):
  global total, right, left, both, eye_l, eye_r, eye_b, check
  if check:
    if pred_l < 0.25 and pred_r < 0.25:
      eye_b = False
    elif eye_b == False and pred_l > 0.75 and pred_r > 0.75:
      both += 1
      total += 1
      eye_b = True
      eye_l = True
      eye_r = True
      print("Both Eye Blinked")
    elif pred_l < 0.25 or pred_r < 0.25:
      if pred_l < 0.25:
        eye_l = False
        eye_b = True

      if pred_r < 0.25:
        eye_r = False
        eye_b = True
    elif pred_l > 0.75 or pred_r > 0.75:
      if eye_l == False and pred_l > 0.75:
        eye_l = True
        eye_b = True
        left += 1
        total += 1
        print("Left Eye Blinked")

      if eye_r == False and pred_r > 0.75:
        eye_r = True
        eye_b = True
        right += 1
        total += 1
        print("Right Eye Blinked")

# main
cap = cv2.VideoCapture("video/WIN_20220609_12_24_33_Pro.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
print((frame_height, frame_width))
out = cv2.VideoWriter('outpy.avi', fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

  img_ori = cv2.resize(img_ori, (1280,720), fx=0.5, fy=0.5)
  print((img_ori.shape[0], img_ori.shape[1]))
  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    cv2.imshow('l', eye_img_l)
    cv2.imshow('r', eye_img_r)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)
    print(pred_l)
    print(pred_r)
    checkBlink(pred_l, pred_r)

    # visualize
    state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
    state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    #cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    #cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    #cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    #cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    text = "Total blinks: " + str(total)
    text2 = "Left blink: " + str(left) + "; Right blink: " + str(right) + "; Both: " + str(both)
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 3, 252), cv2.LINE_4)
    cv2.putText(img, text2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 3, 252), cv2.LINE_4)

    out.write(img)
    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
      break
    elif cv2.waitKey(1) == ord('r'):
      if check:
        print("true")
        check = False
      else:
        print("false")
        check = True
cap.release()
out.release()
cv2.destroyAllWindows()