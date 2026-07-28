from pioneer_sdk2 import Camera, ImageViewer,ServoCamera

camera = Camera()
viewer = ImageViewer()
servo_camera = ServoCamera()



result = servo_camera.set_angle(-80)
print("Трансляция началась")
try:
    while True:
        frame = camera.get_cv_frame(timeout=5.0)

        if frame is not None:
            viewer.imshow("camera", frame, fps=30)

except KeyboardInterrupt:
    print("Трансляция остановлена")

finally:
    viewer.close()
    camera.stop()