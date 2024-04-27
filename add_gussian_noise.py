import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 

### get current work directory
current_dir = os.path.dirname(os.path.abspath(__file__))

#data_loader = Load_data(current_dir+'/HARDataset/')
print(current_dir)
# Load an image
image = cv2.imread(current_dir+'/image.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)

gauss_noise = np.zeros_like(gray_image, np.float32)
cv2.randn(gauss_noise, 0, 1)
gauss_noise = (gauss_noise).astype(np.float32)
gray_image =(gray_image).astype(np.float32)

print(gauss_noise) 
print(gray_image)

scale_factor_gray = np.sqrt(0.02)
scale_factor_noise = np.sqrt(1 - 0.02)

scale_factor_gray2 = np.sqrt(0.01)
scale_factor_noise2 = np.sqrt(1 - 0.01)

scale_factor_gray3 = np.sqrt(0.5)
scale_factor_noise3 = np.sqrt(1 - 0.5)

scale_factor_gray4 = np.sqrt(0.1)
scale_factor_noise4 = np.sqrt(1 - 0.1)

scale_factor_gray5 = np.sqrt(0.25)
scale_factor_noise5 = np.sqrt(1 - 0.25)

# Combine the scaled gray image and scaled Gaussian noise
gn_img_t1 = cv2.add(scale_factor_gray * gray_image, scale_factor_noise * gauss_noise)
gn_img_t1 = (gn_img_t1).astype(np.float32)

gn_img_t2 = cv2.add(scale_factor_gray2 * gn_img_t1, scale_factor_noise2 * gauss_noise)
gn_img_t2 = (gn_img_t1).astype(np.float32)

gn_img_t3= cv2.add(scale_factor_gray3 * gn_img_t2, scale_factor_noise3 * gauss_noise)
gn_img_t3 = (gn_img_t3).astype(np.float32)

gn_img_t4= cv2.add(scale_factor_gray4 * gn_img_t3, scale_factor_noise4 * gauss_noise)
gn_img_t4 = (gn_img_t4).astype(np.float32)

gn_img_t5= cv2.add(scale_factor_gray5 * gn_img_t4, scale_factor_noise5 * gauss_noise)
gn_img_t5 = (gn_img_t5).astype(np.float32)

#denoised_image = cv2.GaussianBlur(gn_img_t5, (5,5), 0)

#denoise
gn_img_inv_t4 = (gn_img_t5 - scale_factor_noise5 * gauss_noise)/scale_factor_gray5
gn_img_inv_t3 = (gn_img_t4 - scale_factor_noise4 * gauss_noise)/scale_factor_gray4
gn_img_inv_t2 = (gn_img_t3 - scale_factor_noise3 * gauss_noise)/scale_factor_gray3
gn_img_inv_t1 = (gn_img_t2 - scale_factor_noise2 * gauss_noise)/scale_factor_gray2
gn_img_inv_t0 = (gn_img_t1 - scale_factor_noise * gauss_noise)/scale_factor_gray

fig=plt.figure()

fig.add_subplot(2,7,1)
plt.imshow(gray_image,cmap='gray')
plt.axis("off")
plt.title("t0")


fig.add_subplot(2,7,2)
plt.imshow(gauss_noise,cmap='gray')
plt.axis("off")
plt.title("noise")

fig.add_subplot(2,7,3)
plt.imshow(gn_img_t1,cmap='gray')
plt.axis("off")
plt.title("t1")


fig.add_subplot(2,7,4)
plt.imshow(gn_img_t2,cmap='gray')
plt.axis("off")
plt.title("t2")

fig.add_subplot(2,7,5)
plt.imshow(gn_img_t3,cmap='gray')
plt.axis("off")
plt.title("t3")

fig.add_subplot(2,7,6)
plt.imshow(gn_img_t4,cmap='gray')
plt.axis("off")
plt.title("t4")


fig.add_subplot(2,7,7)
plt.imshow(gn_img_t5,cmap='gray')
plt.axis("off")
plt.title("t5")

fig.add_subplot(1,7,1)
plt.imshow(gn_img_t5,cmap='gray')
plt.axis("off")
plt.title("t5")

fig.add_subplot(1,7,2)
plt.imshow(gauss_noise,cmap='gray')
plt.axis("off")
plt.title("noise")


fig.add_subplot(1,7,3)
plt.imshow(gn_img_inv_t4,cmap='gray')
plt.axis("off")
plt.title("inv_t4")

fig.add_subplot(1,7,4)
plt.imshow(gn_img_inv_t3,cmap='gray')
plt.axis("off")
plt.title("inv_t3")

fig.add_subplot(1,7,5)
plt.imshow(gn_img_inv_t2,cmap='gray')
plt.axis("off")
plt.title("inv_t2")

fig.add_subplot(1,7,6)
plt.imshow(gn_img_inv_t1,cmap='gray')
plt.axis("off")
plt.title("inv_t1")

fig.add_subplot(1,7,7)
plt.imshow(gn_img_inv_t0,cmap='gray')
plt.axis("off")
plt.title("inv_t0")

plt.show()





