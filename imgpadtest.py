import cv2
img = cv2.imread("E:\\PyProject\\SR\\CodeFormer\\inputs\\test\\1.jpg")
h,w = img.shape[:2]
print(h,w)
if max(img.shape[:2]) < 512:
    ret = cv2.copyMakeBorder(img, 0, 512-h, 0, 512-w, cv2.BORDER_CONSTANT, value=(1,1,1))
elif h<512:
    ret = cv2.copyMakeBorder(img, 0, 512-h, 0, 0, cv2.BORDER_CONSTANT, value=(1,1,1))
elif w<512:
    ret = cv2.copyMakeBorder(img, 0, 0, 0, 512-w, cv2.BORDER_CONSTANT, value=(1,1,1))

out1 = cv2.resize(ret, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
out = out1[:h*2,:w*2,:]
print(out1.shape)
print(out.shape)
cv2.imshow("BORDER_REPLICATE", out1)
cv2.imshow("BORDER_REPLICATE", out)

# 阻塞等待
key = cv2.waitKey(0)