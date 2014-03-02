import zbar
import os
import time
import mpc_control
import numpy

scanner = zbar.ImageScanner()
scanner.parse_config('enable')
from PIL import Image
import io
import time
import picamera

width = 640 #1440
height = 480 #1080
stream = io.BytesIO()
# Capture the image in raw YUV format
with picamera.PiCamera() as camera:
    camera.resolution = (width, height)
    camera.raw_format = 'yuv'
    #camera.start_preview()
    while True:
        camera.capture(stream, 'raw')
        # Rewind the stream for reading
        stream.seek(0)
        image = Image.frombuffer('L', (width,height), stream.getvalue()[0:width*height])
        image.save("frame.png", "PNG")
        exit(1)
        # wrap image data
        #picamera supports raw yuv420 http://picamera.readthedocs.org/en/latest/recipes2.html
        #http://sourceforge.net/apps/mediawiki/zbar/index.php?title=Supported_image_formats zbar supports these formats:
        #422P, I420, YU12, YV12, 411P, YVU9, YUV9, NV12, NV21
#http://www.fourcc.org/yuv.php#Y800
        #image = zbar.Image(width, height, 'Y420', stream.getvalue())
        image = zbar.Image(width, height, 'Y800', stream.getvalue()[0:width*height])
#        import ipdb; ipdb.set_trace()
#        image.write("zbar.raw")

        # scan the image for barcodes
        print("scan")
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
