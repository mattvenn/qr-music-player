# QR music player

Based on an idea from [Dave Murray-Rust](http://www.mo-seph.com/), this is a set of small scripts that lets you get 'hands on' with your digital music collection.

Print out your album art on one side and a version 1 (smallest size) QR code on the other. 

A webcam on your computer/Raspberry Pi picks up the QR code using zlib and starts the album playing.

# Scripts

## cam_qr

* Starts the webcam, uses PIl to convert an image from open cv format to something zlib understands, 
* Uses zlib to search for qr codes,
* If one is found, search database and pass album name to mpc.

## mpc control

* Defines a single function that will start a new album playing if it isn't already

## make_qr

* Creates a new qr code using qrcode python module

# Things to make it better

* a real database, and way of generating it,
* auto generate pdfs with qr and album art for printing,
* extra qr codes to stop and start,
* chop off bottom of image to make scanning quicker,
* other music controllers (currently only works with mpc),

# Pre requisites

* sudo apt-get install python-zbar
* python opencv (cv2) - maybe we can avoid this?
* python (pil) Image - can avoid this too?
* mpc and mpd for music playing
* python qrcode
