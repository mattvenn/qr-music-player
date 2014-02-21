import zbar
from PIL import Image

scanner = zbar.ImageScanner()
scanner.parse_config('enable')
pil = Image.open("webcam.png").convert('L')
width, height = pil.size
raw = pil.tostring()

# wrap image data
image = zbar.Image(width, height, 'Y800', raw)

# scan the image for barcodes
scanner.scan(image)

for symbol in image:
    print symbol.data
