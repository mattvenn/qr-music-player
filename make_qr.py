import qrcode
qr = qrcode.QRCode(
    version = 1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=50,
    border=4,
)
import random
id = random.randint(0,1024)
print( id)
qr.add_data(id)
qr.make()
img = qr.make_image()
img.save("qr.png")
