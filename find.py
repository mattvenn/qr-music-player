import numpy as np
import cv2
import glob

# create video capture
cap = cv2.VideoCapture(0)
h = 480
w = 640
cap.set(3,w)
cap.set(4,h)

SHADOW_V = 150 #the (hs)v threshold for up or down.
SIFT = False
ORB = True

real_width = 30
#load fiducials
fids = glob.glob('fiducials/*.png')
tags = []
count = 0
for fid in fids:
    tags.append({
        'img' : cv2.imread(fid,0),
        'id' : count,
        'file' : fid,
        })
     
    count +=1 
# h and w of fiducial
h,w = tags[0]['img'].shape

#braille pin spacing and positioning wrt top left corner of fiducial
x_pitch = 2.5
y_pitch = 2.5
pin_x = 40.0 + x_pitch / 4
pin_y = 10.0 + y_pitch / 4
shadow_size = 10

#useful function
def mm2px(mm):
    return mm * (w/real_width)

#load in one of our distorted images
#img = cv2.imread('photo.jpg') # trainImage

# Load previously saved data about the camera - generated with the chessboard photos
try:
    calib = np.load('B.npz')
    mtx = calib['mtx']
    dist = calib['dist']

    #undistort the image!
    ph, pw = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(pw,ph),1,(pw,ph))
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)
except IOError as e:
    print("couldn't open camera file, not undistorting...")

if ORB:
    MIN_MATCH_COUNT = 80
    # Initiate ORB detector
    print("using ORB")
    orb = cv2.ORB()
elif SIFT:
    MIN_MATCH_COUNT = 8
    # Initiate SIFT detector
    print("using SIFT")
    sift = cv2.SIFT()

#find the keypoints and descriptors with ORB
#fiducial
for tag in tags:
    print("processing fid %d %s" % (tag['id'], tag['file']))
    if ORB:
        tag['kp'] = orb.detect(tag['img'],None)
        tag['kp'], tag['des'] = orb.compute(tag['img'],tag['kp'])
    elif SIFT:
        # find the keypoints and descriptors with SIFT
        tag['kp'],tag['des'] = sift.detectAndCompute(tag['img'],None)

def compute_image(img):
#    print("processing image")
    if ORB:
        #photo
        img_kp = orb.detect(img,None)
        img_kp, img_des = orb.compute(img,img_kp)
    elif SIFT:
        #photo
        img_kp, img_des = sift.detectAndCompute(img,None)
        #print img_kp
        #print img_des
    return(img_des,img_kp)

print("done")
#does the match, if it's good returns the homography transform
def find(des,kp,img_des,img_kp):
    if ORB:
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des,img_des)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)[:MIN_MATCH_COUNT]
    elif SIFT:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des,img_des,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        matches = good


#    print "matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT)
    
    if len(matches)>=MIN_MATCH_COUNT:
        src_pts = np.float32([ kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ img_kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        #get the transformation between the flat fiducial and the found fiducial in the photo
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        #return the transform
        return( M,len(matches))
    else:
#        print "Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT)
        return (None,len(matches))

#draws boxes round the important things: fiducial, pins
def draw_outlines(M,img):
    #array containing co-ords of the fiducial
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #transform the coords of the fiducial onto the picture
    dst = cv2.perspectiveTransform(pts,M)
    #draw a box around the fiducial
    cv2.polylines(img,[np.int32(dst)],True,(255,0,0),5, cv2.CV_AA)

    #repeat for the pins
    pts=np.float32([
        [mm2px(pin_x),mm2px(pin_y)],
        [mm2px(pin_x+2*x_pitch),mm2px(pin_y)],
        [mm2px(pin_x+2*x_pitch),mm2px(pin_y+3*y_pitch)],
        [mm2px(pin_x),mm2px(pin_y+3*y_pitch)],
        ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    cv2.polylines(img,[np.int32(dst)],True,(0,255,0),5, cv2.CV_AA)

#find the shadows of the pins!
def detect_shadows(M):
    pin_num = 0
    #for each shadow: pins are numbered 0 to 2 in the left column, 3 to 5 in the right
    for x in range(2):
        for y in range(3):

            pin_pt=np.float32([
                [mm2px(pin_x+x*x_pitch),mm2px(pin_y+y*y_pitch)],
                ]).reshape(-1,1,2)
            #transform the pin coords
            pin_dst = cv2.perspectiveTransform(pin_pt,M)

            #the pin x and y in the image
            pin_x_dst = pin_dst[0][0][0]
            pin_y_dst = pin_dst[0][0][1]
            #print pin_x_dst , pin_y_dst

            #define a ROI for the shadow - cv2 only does square
            roi = img[pin_y_dst:pin_y_dst+shadow_size,pin_x_dst:pin_x_dst+shadow_size]

            #save it for reference
            cv2.imwrite('roi' + str(pin_num) + '.png', roi)

            #convert to HSV
            hsvroi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

            #mean value
            val = cv2.mean(hsvroi)[2]

            #is it up or down?
            print("pin %d v%0.2f = %s" % (pin_num, val, "down" if val > SHADOW_V else "up"))
            pin_num += 1

    #cv2.circle(img,(int(pin_x_dst),int(pin_y_dst)),10,(255,0,0),-1)
    #put a box round the pins
    #roidst = cv2.perspectiveTransform(roipts,M)
    #cv2.polylines(img,[np.int32(roidst)],True,(255,0,0),3, cv2.CV_AA)
    #cv2.line(img,tuple(roidst[0][0]),tuple(roidst[1][0]),(255,0,0),3)


#find the fiducial
print( "loop...")
while True:
    success,frame = cap.read()
    if success:
        (img_des,img_kp) = compute_image(frame)
        if len(img_kp):

            for tag in tags:
                (M,matches) = find(tag['des'],tag['kp'],img_des,img_kp)
                if M != None:
                    print("tag %d : %d" %( tag['id'],matches))
                    #detect_shadows(M)
                    draw_outlines(M,frame)
                    """
                    #write out full size image
                    cv2.imwrite('found.png', img)
                    """
                    #resize for display
    #                newx,newy = img.shape[1]/4,img.shape[0]/4 #new size (w,h)
    #                smallimg = cv2.resize(img,(newx,newy))

        cv2.imshow('found',frame)
        if cv2.waitKey(33)== 27:
            break
            break
