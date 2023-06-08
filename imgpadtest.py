# import cv2
# img = cv2.imread("E:\\\\PyProject\\\\SR\\\\CodeFormer\\\\inputs\\\\test\\\\1.jpg")
# h,w = img.shape[:2]
# print(h,w)
# if max(img.shape[:2]) < 512:
#     ret = cv2.copyMakeBorder(img, 0, 512-h, 0, 512-w, cv2.BORDER_CONSTANT, value=(1,1,1))
# elif h<512:
#     ret = cv2.copyMakeBorder(img, 0, 512-h, 0, 0, cv2.BORDER_CONSTANT, value=(1,1,1))
# elif w<512:
#     ret = cv2.copyMakeBorder(img, 0, 0, 0, 512-w, cv2.BORDER_CONSTANT, value=(1,1,1))

# out1 = cv2.resize(ret, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
# out = out1[:h*2,:w*2,:]
# print(out1.shape)
# print(out.shape)
# cv2.imshow("BORDER_REPLICATE", out1)
# cv2.imshow("BORDER_REPLICATE", out)

# # 阻塞等待
# key = cv2.waitKey(0)
import numpy as np
import cv2
import os

# 读取图片
img_dir = 'E:\\PyProject\\data\\classical_SR_datasets\\mohulei\\haibao\\haibaosr'
folder_path = 'E:\\PyProject\\data\\classical_SR_datasets\\mohulei\\haibao\\haibaosr\\SwinIR-Codeformer\\results\\ori_0.9\\final_results'
ori_dir = 'E:\\PyProject\\data\\classical_SR_datasets\\mohulei\\haibao\\haibaosr\\ori'
import os

# 定义文件夹路径和匹配字符串
match_str = 'SwinSR_Codeformer0.9_'

# 遍历文件夹
# for file_name in os.listdir(folder_path):
#     # 判断文件名是否包含匹配字符串
#     if match_str in file_name:
#         # 构造新文件名
#         new_file_name = file_name.replace(match_str, '')
#         # 重命名文件
#         os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))

ori = os.listdir(ori_dir)
bad_img_list = os.listdir(img_dir)
sr_img_list = os.listdir(folder_path)
imgs = []
for img_name in ori:
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        ori_img_path = os.path.join(ori_dir, img_name)
        bad_img_path = os.path.join(img_dir, img_name)
        sr_img_path = os.path.join(folder_path, img_name)
        bad_img = cv2.imread(bad_img_path)
        ori_img = cv2.imread(ori_img_path)
        ori_img = cv2.resize(ori_img, (bad_img.shape[1], bad_img.shape[0]))
        print(bad_img.shape[0], bad_img.shape[1])

        sr_img = cv2.imread(sr_img_path)
        print(bad_img.shape)
        print(ori_img.shape)
        print(sr_img.shape)
        imgs.append(ori_img)
        imgs.append(bad_img)
        imgs.append(sr_img)
        cv2.imshow("BORDER_REPLICATE", sr_img)

        # 阻塞等待
        key = cv2.waitKey(0)
    break
# 将图片拼接成一列
img_col = np.hstack(imgs)

# 显示拼接后的图片
cv2.imshow('Images', img_col)
cv2.waitKey(0)
cv2.destroyAllWindows()
