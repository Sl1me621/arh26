from pioneer_sdk2 import Camera, ImageViewer
import cv2

camera = Camera()
viewer = ImageViewer()
frame = camera.get_cv_frame()

height, width = frame.shape[:2]

writer = cv2.VideoWriter(
    "flight.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (width, height)
)

while True:
    frame = camera.get_cv_frame()

    """
    Здесь располагается ваш код 
    для автономного полета и обработки изображения
    """
    
    writer.write(frame)
    viewer.imshow("video", frame)