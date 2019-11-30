import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, color
# skimage.io skimage.color

# im1 = plt.imread('chessboard.png', CV_LOAD_IMAGE_GRAYSCALE)
# tmpim1 = cv2.imread('chessboard.png')
image1 = io.imread('chessboard.png')
im1 = color.rgb2gray(image1)
image2 = io.imread('jellyfish.jpg')
im2 = color.rgb2gray(image2)
image3 = io.imread('new_york.jpg')
im3 = color.rgb2gray(image3)

#Normalized Data to get it between 0 and 1
# x = im1.ravel()
# normalized = (x-min(x))/(max(x)-min(x))
# y = im2.ravel()
# normalized = (y-min(y))/(max(y)-min(y))
# z = im3.ravel()
# normalized = (z-min(z))/(max(z)-min(z))
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

out1 = im2double(im1) # Convert to normalized floating point
out2 = im2double(im2) # Convert to normalized floating point
out3 = im2double(im3) # Convert to normalized floating point
# plt.imshow(out1, cmap='gray')
# plt.show()
# plt.imshow(out2, cmap='gray')
# plt.show()
# plt.imshow(out3, cmap='gray')
# plt.show()


u1,s1,v1 = np.linalg.svd(out1)
# newa = np.array(s)
logs1 = np.log(s1)
u2,s2,v2 = np.linalg.svd(out2)
logs2 = np.log(s2)
u3,s3,v3 = np.linalg.svd(out3)
logs3 = np.log(s3)

plt.plot(logs1)
plt.title('log of singular values of chessboard')
plt.savefig('log_chessboard.jpg')
plt.show()
plt.plot(logs2)
plt.title('log of singular values of jellyfish')
plt.savefig('log_jellyfish.jpg')
plt.show()
plt.plot(logs3)
plt.title('log of singular values of new york')
plt.savefig('log_new_york.jpg')
plt.show()


def find_r(m,n,s):
    r = np.floor((m*n)/(m+n+1))
    print(r)
    x = np.log(s)
    r = int(r)
    counter = 0
    for i in range(len(x)):
        if x[i] > 0:
            counter+=1
    print(counter)
    return min(r, counter)
    # return r

m,n = np.shape(im1)
print(m,n)
# r = find_r(m,n,s1)
r = 2
# s1 = np.array(s1)
s1 = np.diag(s1)
print(s1)
news1 = s1[:r,:r]

print(news1)
newu1 = u1[:,:r]
newv1 = v1[:r,:]
compressedimage1_part1 = np.dot(newu1, news1)
compressedimage1 = np.dot(compressedimage1_part1, newv1)
plt.imshow(compressedimage1, cmap='gray')
plt.title('compressed chessboard with r=2')
plt.savefig('chessboard_compressed.jpg')
plt.show()

# image 2
m,n = np.shape(im2)
print(m,n)
# r = find_r(m,n,s2)
r = 40
s2 = np.diag(s2)
news2 = s2[:r,:r]
newu2 = u2[:,:r]
newv2 = v2[:r,:]
compressedimage2_part1 = np.dot(newu2, news2)
compressedimage2 = np.dot(compressedimage2_part1, newv2)
plt.imshow(compressedimage2, cmap='gray')
plt.title('compressed jellyfish with r=40')
plt.savefig('jellyfish_compressed.jpg')
plt.show()

# image 3
m,n = np.shape(im3)
print(m,n)
# r = find_r(m,n,s3)
r = 350
print(r)
s3 = np.diag(s3)
news3 = s3[:r,:r]
newu3 = u3[:,:r]
newv3 = v3[:r,:]
compressedimage3_part1 = np.dot(newu3, news3)
compressedimage3 = np.dot(compressedimage3_part1, newv3)
plt.imshow(compressedimage3, cmap='gray')
plt.title('compressed new york with r=350')
plt.savefig('new_york_compressed.jpg')
plt.show()
# size(u*v*s) = m*r+r+r*n

# print(compressedratio1)
# compressedratio1 = (m*n)/((m+n+1)*r)
# compressedratio2 = (m*n)/((m+n+1)*r)
# compressedratio3 = (m*n)/((m+n+1)*r)
