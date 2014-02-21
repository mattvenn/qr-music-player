import Image, cv2
import zbar
import os
import time
import mpc_control

scanner = zbar.ImageScanner()
scanner.parse_config('enable')

cap = cv2.VideoCapture(0)
h = 640
w = 480
cap.set(3,w)
cap.set(4,h)

while True:
    _,cv2_im = cap.read()
    cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im).convert('L')
    width,height= pil_im.size
    # wrap image data
    image = zbar.Image(width, height, 'Y800', pil_im.tostring())
    # scan the image for barcodes
    scanner.scan(image)
    for symbol in image:
        print( symbol.data)
        if symbol.data == '429':
            album = "Incunabula"
            mpc_control.add(album)
        elif symbol.data == '767':
            album = "Legend"
            mpc_control.add(album)

#    cv2.imshow("web",cv2_im)
#    if cv2.waitKey(33)== 27:
#        pil_im.save("webcam.png")
#        break
