# %%
import cv2
import cv2.aruco as aruco
import numpy as np
import math

# %%
aruco_dict = {} #to store rotated aruco markers

# %%
def findAruco(img): #to identify aruco marker ids and rotate to relevant angles
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoPara = aruco.DetectorParameters_create()

    (corners,ids,rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters = arucoPara)

    print(ids)
    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner,markerid) in zip(corners,ids):
            corners = markerCorner.reshape((4,2))
            (topLeft,topRight,bottomLeft,bottomRight) = corners

            topRight = (int(topRight[0]),int(topRight[1]))
            topLeft = (int(topLeft[0]),int(topLeft[1])) 
            bottomLeft = (int(bottomLeft[0]),int(bottomLeft[1]))
            bottomRight = (int(bottomRight[0]),int(bottomRight[1]))

            cx = int((topLeft[0]+bottomRight[0])/2.0) #centre coordinates
            cy = int((topLeft[1]+bottomRight[1])/2.0)

            cx1 = int((topLeft[0]+bottomLeft[0])/2.0) #centre of one edge
            cy1 = int((topLeft[1]+bottomLeft[1])/2.0)

            imgcopy = img.copy()

            cv2.circle(imgcopy, (cx,cy),5,(255,0,0),-1)
            cv2.circle(imgcopy, (cx1,cy1),5,(255,0,0),-1)
            cv2.line(imgcopy,(cx,cy),(cx1,cy1),(255,0,0),2)

            slope = (cy1-cy)/(cx1-cx)
            angle = (math.atan(slope)) * 180 / math.pi
            cv2.putText(imgcopy,str(angle),(cx,cy+10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            cv2.imshow("Display angle",imgcopy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Generating a rotation matrix 

            matrix = cv2.getRotationMatrix2D((cx,cy), angle, 1.0) 
            h,w,c = img.shape

            # Performing the affine transformation 

            rotated = cv2.warpAffine(img, matrix, (w, h))

        
            aruco_dict[markerid] = rotated
            
            cv2.imshow("Rotated image",rotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# %%
img_a = cv2.imread(r"C:\Users\HP\OneDrive\Desktop\OpenCV_Project_Spriha\Ha.jpg")
img_b = cv2.imread(r"C:\Users\HP\OneDrive\Desktop\OpenCV_Project_Spriha\HaHa.jpg")
img_c = cv2.imread(r"C:\Users\HP\OneDrive\Desktop\OpenCV_Project_Spriha\LMAO.jpg")
img_d = cv2.imread(r"C:\Users\HP\OneDrive\Desktop\OpenCV_Project_Spriha\XD.jpg")

findAruco(img_a)
findAruco(img_b)
findAruco(img_c)
findAruco(img_d)

# %%
#Crop aruco markers to remove extra boundary area

cropped_aruco_1 = aruco_dict[1][150:570,105:515]
cropped_aruco_2 = aruco_dict[2][13:478,70:535]
cropped_aruco_3 = aruco_dict[3][77:515,77:515]
cropped_aruco_4 = aruco_dict[4][130:566,85:521]

cropped_aruco_dict = {1:cropped_aruco_1,2:cropped_aruco_2,3:cropped_aruco_3,4:cropped_aruco_4}

cv2.imshow("cropped_1",cropped_aruco_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("cropped_2",cropped_aruco_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("cropped_3",cropped_aruco_3)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("cropped_4",cropped_aruco_4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
#get the corner coordinates of the cropped aruco markers
aruco_coord_dict = {}

for i in range(1,5):
   
    coordinates = [[0, cropped_aruco_dict[i].shape[1]], [0, 0], [cropped_aruco_dict[i].shape[1], 0],[cropped_aruco_dict[i].shape[1], cropped_aruco_dict[i].shape[0]]]
    aruco_coord_dict[i] = coordinates

aruco_coord_dict[2] = [[0, 0], [cropped_aruco_dict[i].shape[1], 0],[cropped_aruco_dict[i].shape[1], cropped_aruco_dict[i].shape[0]], [0, cropped_aruco_dict[i].shape[1]]]

# %%
#Open the initial task image and resize it
img1 = cv2.imread(r"C:\Users\HP\OneDrive\Desktop\OpenCV_Project_Spriha\CVtask.jpg")
h,w,c = img1.shape
print("Original Width and Height:", w,"x", h)

resized_img = cv2.resize(img1, (877,620))
cv2.imshow("Original",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
colour_coord_dict = {} #to store corner coordinates of all squares of different colours
width_height = {} #to store the width and height of the square boxes

# %%
#to detect squares among similar coloured shapes
#lower_bound and upper_bound specifies the rgb range of colour

def detect_Coloured_Squares(colour,lower_bound,upper_bound):

    mask = cv2.inRange(resized_img, lower_bound, upper_bound)

    kernel = np.ones((7,7),np.uint8)
    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmented_img = cv2.bitwise_and(resized_img, resized_img, mask=mask) #only the specified coloured portions in the image are displayed

    img2 = segmented_img.copy()

    l = 4
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,peri*0.04,True)
            if len(approx) == l:
            
                x,y,w,h = cv2.boundingRect(cnt)
                ar = w / float(h)

                if ar >= 0.95 and ar <= 1.05:  #conditions for square image

                    rect = cv2.minAreaRect(cnt) #to bound in a particular angle
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img2,[box],-1,(255,0,0),4)
                    cv2.putText(img2,f" {len(cnt)}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,-1,(0,0,0),2)
                    #cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)
                    print(rect,box)

                    lst_coord = []    #get the x and y coordinates for all four corners

                    lst_coord_0 = [list(box)[0][0],list(box)[0][1]] 
                    lst_coord.append(lst_coord_0)

                    lst_coord_1 = [list(box)[1][0],list(box)[1][1]]
                    lst_coord.append(lst_coord_1)

                    lst_coord_2 = [list(box)[2][0],list(box)[2][1]]
                    lst_coord.append(lst_coord_2)

                    lst_coord_3 = [list(box)[3][0],list(box)[3][1]]
                    lst_coord.append(lst_coord_3)
                   
                    print("Corner points of box: ",lst_coord)
                    print("Width and height: ",rect[1])
                    
                    colour_coord_dict[colour] = lst_coord
                    width_height[colour] = rect[1]

    cv2.imshow("image_contours",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

# %%
# lower bound and upper bound for Green color
colour_green = "green"
lower_bound_green = np.array([60, 60, 60])   
upper_bound_green = np.array([100, 255, 255])

detect_Coloured_Squares(colour_green,lower_bound_green,upper_bound_green)


# %%
# lower bound and upper bound for Orange color
colour_orange = "orange"
lower_bound_orange = np.array([1,20,20])   
upper_bound_orange = np.array([18, 255, 255])

detect_Coloured_Squares(colour_orange,lower_bound_orange,upper_bound_orange)

# %%
# lower bound and upper bound for Black color
colour_black = "black"
lower_bound_black = np.array([0, 0, 0])   
upper_bound_black = np.array([1, 1, 1])

detect_Coloured_Squares(colour_black,lower_bound_black,upper_bound_black)

# %%
# lower bound and upper bound for Pink Peach color
colour_pinkpeach = "pink-peach"
lower_bound_pinkpeach = np.array([200, 200, 200])   
upper_bound_pinkpeach = np.array([245, 245, 245])

detect_Coloured_Squares(colour_pinkpeach,lower_bound_pinkpeach,upper_bound_pinkpeach)

# %%
final_parts = {} #stores images of all the aruco markers pasted one at a time

# %%
def place_aruco(j,colour):
    pts_src = np.array(aruco_coord_dict[j]) #coordinates of all four corners of the aruco markers

    pts_dst = np.array(colour_coord_dict[colour]) #roi corresponding to the specific colour according to marker id

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(cropped_aruco_dict[j], h, (resized_img.shape[1],resized_img.shape[0])) #arucomarker placed on the specified coordinates

    final_parts[j] = im_out

    cv2.imshow("Warped Source Image", im_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %%
place_aruco(1,"green")
place_aruco(2,"orange")
place_aruco(3,"black")
place_aruco(4,"pink-peach")

# %%
#roi areas are turned black(pixels=0) 
resized_img = cv2.fillConvexPoly(resized_img, np.array(colour_coord_dict["green"]), (0,0,0))
resized_img = cv2.fillConvexPoly(resized_img, np.array(colour_coord_dict["orange"]), (0,0,0))
resized_img = cv2.fillConvexPoly(resized_img, np.array(colour_coord_dict["black"]), (0,0,0))
resized_img = cv2.fillConvexPoly(resized_img, np.array(colour_coord_dict["pink-peach"]), (0,0,0))

for i in range(1,5):

    newmask = np.array(np.zeros(resized_img.shape))
    newmask = cv2.fillConvexPoly(newmask,np.array(aruco_coord_dict[i]),(255,255,255))

    kernel1 = np.ones((7,7),np.uint8)
    newmask = cv2.morphologyEx(newmask, cv2.MORPH_CLOSE, kernel1)
    newmask = cv2.morphologyEx(newmask, cv2.MORPH_OPEN, kernel1)

    th = cv2.bitwise_and(final_parts[i],final_parts[i], newmask) #combining the mask with the task image

    resized_img += final_parts[i]   #the final_parts are put together in one final image
    cv2.imshow("placed_aruco",resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %%
cv2.imshow("final",resized_img)

cv2.imwrite("Final.jpg",resized_img) #the final image file is saved
cv2.waitKey(0)
cv2.destroyAllWindows()


