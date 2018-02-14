from PIL import Image
import time
import cv2
import zbar
import os
import time
import mpc_control
from led_control import LEDControl

led = LEDControl()
led.set_color(0, 255, 0)

scanner = zbar.ImageScanner()
scanner.parse_config('enable')

cap = cv2.VideoCapture(0)
h = 640
w = 480
cap.set(3,w)
cap.set(4,h)

# none of the get stuff works due to webcam not being properly supported in v4l ?
#frameRate = cap.get(cv2.CAP_PROP_FPS) #frame rate

last_time = 0
scan_interval = 2 # second

while True:
    _,cv2_im = cap.read()
    cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im).convert('L')
    width,height= pil_im.size
    # wrap image data
    image = zbar.Image(width, height, 'Y800', pil_im.tostring())

    # only bother scanning if some time has gone by since last scan
    if time.time() > last_time + scan_interval:
        # scan the image for barcodes
        scanner.scan(image)
        for symbol in image:
            last_time = time.time()
            print( symbol.data)
            pil_im.save("webcam.png")
            led.set_color(0, 0, 255)
            time.sleep(0.5)
            if symbol.data == '429':
                album = "staysitting"
                mpc_control.add(album)
            elif symbol.data == '767':
                album = "sitdown"
                mpc_control.add(album)
            else:
                led.set_color(255, 0, 0)

            time.sleep(0.5)
            led.set_color(0, 255, 0)

#    cv2.imshow("web",cv2_im)
#    if cv2.waitKey(33)== 27:
#        pil_im.save("webcam.png")
#        break
