from PIL import Image, ImageDraw

width = 500

border_1 = 20
border_2 = 40
border_3 = 80

black = (255,255,255)
white = (0,0,0)

size = 4
sq_size = (width-border_3*2)/size

#create an image
im = Image.new("RGB", (width,width), white)
#get the draw object
bits = [
    [ 1,0,1,1 ],
    [ 0,0,0,1 ],
    [ 1,0,0,0 ],
    [ 1,1,0,0 ],
    ]
draw = ImageDraw.Draw(im)
#draw.rectangle([border_1,border_1,width-border_1,width-border_1],fill=black)
#draw.rectangle([border_2,border_2,width-border_2,width-border_2],fill=white)
#draw.rectangle(coords,fill=colour)
for x in range(size):
    for y in range(size):
        bit = bits[y][x]
        coords=[border_3+x*sq_size,border_3+y*sq_size,border_3+x*sq_size+sq_size,border_3+y*sq_size+sq_size]
        if bit:
            draw.rectangle(coords,fill=black)
        if (x == 0 or x == 3 )and y == 0 :
            coords=[border_3+sq_size/3+x*sq_size,border_3+sq_size/3+y*sq_size,border_3+x*sq_size+sq_size-sq_size/3,border_3+y*sq_size+sq_size-sq_size/3]
            draw.rectangle(coords,fill=white)

#save the image
im.save("test.png", "PNG")


