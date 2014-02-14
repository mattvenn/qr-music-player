import numpy as np
import cv2
import glob
import time

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

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6
def init_feature(name):
    if name == 'sift':
        detector = cv2.SIFT()
        norm = cv2.NORM_L2
    elif name == 'surf':
        detector = cv2.SURF(800)
        norm = cv2.NORM_L2
    elif name == 'orb':
        detector = cv2.ORB(400)
        norm = cv2.NORM_HAMMING
    elif name == 'brisk':
        detector = cv2.BRISK()
        norm = cv2.NORM_HAMMING
    else:
        return None, None

    if norm == cv2.NORM_L2:
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    else:
        flann_params= dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6, # 12
                           key_size = 12,     # 20
                           multi_probe_level = 1) #2
    matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    return detector, matcher


#load fiducials
def init_fiducials(detector):
    fids = glob.glob('fiducials/*.png')
    tags = []
    for fid in fids:
        print("processing fid %d %s" % (len(tags), fid))
        tag = {
            'img' : cv2.imread(fid,0),
            'id' : len(tags),
            'file' : fid,
            }
         
        # h and w of fiducial
        tag['h'],tag['w'] = tag['img'].shape

        tag['kp'], tag['des'] = detector.detectAndCompute(tag['img'],None)
        tag['numkp'] = len(tag['kp'])
        tags.append(tag)
    return tags

def compute_image(img,detector):
    img_kp, img_des = detector.detectAndCompute(img,None)
    return(img_des,img_kp)

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

#does the match, if it's good returns the homography transform
def find(matcher,desc1,desc2,kp1,kp2):
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    #    print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        num = len(status)
    else:
        H, status = None, None
    #    print '%d matches found, not enough for homography estimation' % len(p1)
        num = len(p1)

    return( H, num)

#draws boxes round the important things: fiducial, pins
def draw_outlines(H,img,tag):
    #array containing co-ords of the fiducial
    pts = np.float32([ [0,0],[0,tag['h']-1],[tag['w']-1,tag['h']-1],[tag['w']-1,0] ]).reshape(-1,1,2)
    #transform the coords of the fiducial onto the picture
    dst = cv2.perspectiveTransform(pts,H)
    #draw a box around the fiducial
    cv2.polylines(img,[np.int32(dst)],True,(255,0,0),5, cv2.CV_AA)

    #repeat for the pins
    """
    pts=np.float32([
        [mm2px(pin_x),mm2px(pin_y)],
        [mm2px(pin_x+2*x_pitch),mm2px(pin_y)],
        [mm2px(pin_x+2*x_pitch),mm2px(pin_y+3*y_pitch)],
        [mm2px(pin_x),mm2px(pin_y+3*y_pitch)],
        ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,H)
    cv2.polylines(img,[np.int32(dst)],True,(0,255,0),5, cv2.CV_AA)
    """

"""
#find the shadows of the pins!
def detect_shadows(H):
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
"""

if __name__ == '__main__':
    #find the fiducial
    detector = 'orb'
    print( "starting, using %s", detector)
    detector, matcher = init_feature(detector)
    tags = init_fiducials(detector)
    for tag in tags:
        print( "tag %d, file %s, kps %d" % (tag['id'],tag['file'],tag['numkp']))

    while True:
        success,frame = cap.read()
        if success:
            (img_des,img_kp) = compute_image(frame,detector)
            for tag in tags:
                (H,matches) = find(matcher,tag['des'],img_des,tag['kp'],img_kp)

                print("tag %d : %d" %( tag['id'],matches))
                if matches >= 20:
                    draw_outlines(H,frame,tag)

            cv2.imshow('found',frame)
            if cv2.waitKey(33)== 27:
                break
        time.sleep(0.1)
