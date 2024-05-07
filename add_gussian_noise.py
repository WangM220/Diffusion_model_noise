import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 

# get current work directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Load an image
image = cv2.imread(current_dir+'/image.jpg')

# switch the figure into gray 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)

# produce gaussian noise
gauss_noise = np.zeros_like(gray_image, np.float32)
cv2.randn(gauss_noise, 0, 1)

gauss_noise = (gauss_noise).astype(np.float32)
gray_image =(gray_image).astype(np.float32)

print(gauss_noise) 
print(gray_image)

# set betat1, betat2, betat3, betat4, betat5 as 0.02, 0.01, 0.5, 0.1, 0.25

scale_factor_gray = np.sqrt(0.02)
scale_factor_noise = np.sqrt(1 - 0.02)

scale_factor_gray2 = np.sqrt(0.01)
scale_factor_noise2 = np.sqrt(1 - 0.01)

scale_factor_gray3 = np.sqrt(0.5)
scale_factor_noise3 = np.sqrt(1 - 0.5)

scale_factor_gray4 = np.sqrt(0.1)
scale_factor_noise4 = np.sqrt(1 - 0.1)

scale_factor_gray5 = np.sqrt(0.001)
scale_factor_noise5 = np.sqrt(1 - 0.001)


# add Gaussian noise 
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


# compare the pixel distribution 
pixel_values = gn_img_t1.flatten()
pixel_values_t5 = gn_img_t5.flatten()
pixel_values_noise = gauss_noise.flatten()

### Probability Distribution of Image t0

histogram, bins = np.histogram(pixel_values, bins=256, range=(-5, 255))
print("mean",np.mean(pixel_values))
print("variance", np.var(pixel_values))
# Normalize the histogram to get probability density
probability_density = histogram / np.sum(histogram)

plt.figure(figsize=(8, 6))
plt.bar(bins[:-1], probability_density)
plt.xlabel('Density')
plt.ylabel('t0')
plt.title('Probability Distribution of Image')
plt.show()

### Probability Distribution of Image t5

histogram, bins = np.histogram(pixel_values_t5, bins=256, range=(-5, 255))
print("mean",np.mean(pixel_values_t5))
print("variance", np.var(pixel_values_t5))

probability_density_t5 = histogram / np.sum(histogram)

plt.figure(figsize=(8, 6))
plt.bar(bins[:-1], probability_density_t5)
plt.xlabel('Density')
plt.ylabel('t5')
plt.title('Probability Distribution of Image t5')
plt.show()

### Probability Distribution of Noise

histogram, bins = np.histogram(pixel_values_noise, bins=256, range=(-5, 255))
print("noise mean",np.mean(pixel_values_noise))
print("noise variance", np.var(pixel_values_noise))

probability_density_noise = histogram / np.sum(histogram)

plt.figure(figsize=(8, 6))
plt.bar(bins[:-1], probability_density_noise)
plt.xlabel('Density')
plt.ylabel('Noise')
plt.title('Probability Distribution of Noise')
plt.show()


#denoised_image = cv2.GaussianBlur(gn_img_t5, (5,5), 0)

#denoise
gn_img_inv_t4 = (gn_img_t5 - scale_factor_noise5 * gauss_noise)/scale_factor_gray5
gn_img_inv_t3 = (gn_img_t4 - scale_factor_noise4 * gauss_noise)/scale_factor_gray4
gn_img_inv_t2 = (gn_img_t3 - scale_factor_noise3 * gauss_noise)/scale_factor_gray3
gn_img_inv_t1 = (gn_img_t2 - scale_factor_noise2 * gauss_noise)/scale_factor_gray2
gn_img_inv_t0 = (gn_img_t1 - scale_factor_noise * gauss_noise)/scale_factor_gray

fig=plt.figure(dpi=300)

fig.add_subplot(2,7,1)
plt.imshow(gray_image,cmap='gray')
plt.axis("off")
plt.title("t0",fontsize=5)

fig.add_subplot(2,7,2)
plt.imshow(gauss_noise,cmap='gray')
plt.axis("off")
plt.title("noise",fontsize=5)

fig.add_subplot(2,7,3)
plt.imshow(gn_img_t1,cmap='gray')
plt.axis("off")
plt.title("t1",fontsize=5)

fig.add_subplot(2,7,4)
plt.imshow(gn_img_t2,cmap='gray')
plt.axis("off")
plt.title("t2",fontsize=5)

fig.add_subplot(2,7,5)
plt.imshow(gn_img_t3,cmap='gray')
plt.axis("off")
plt.title("t3",fontsize=5)

fig.add_subplot(2,7,6)
plt.imshow(gn_img_t4,cmap='gray')
plt.axis("off")
plt.title("t4",fontsize=5)

fig.add_subplot(2,7,7)
plt.imshow(gn_img_t5,cmap='gray')
plt.axis("off")
plt.title("t5",fontsize=5)

fig.add_subplot(1,7,1)
plt.imshow(gn_img_t5,cmap='gray')
plt.axis("off")
plt.title("t5",fontsize=5)

fig.add_subplot(1,7,2)
plt.imshow(gauss_noise,cmap='gray')
plt.axis("off")
plt.title("noise",fontsize=5)


fig.add_subplot(1,7,3)
plt.imshow(gn_img_inv_t4,cmap='gray')
plt.axis("off")
plt.title("t4",fontsize=5)

fig.add_subplot(1,7,4)
plt.imshow(gn_img_inv_t3,cmap='gray')
plt.axis("off")
plt.title("t3",fontsize=5)

fig.add_subplot(1,7,5)
plt.imshow(gn_img_inv_t2,cmap='gray')
plt.axis("off")
plt.title("t2",fontsize=5)

fig.add_subplot(1,7,6)
plt.imshow(gn_img_inv_t1,cmap='gray')
plt.axis("off")
plt.title("t1",fontsize=5)

fig.add_subplot(1,7,7)
plt.imshow(gn_img_inv_t0,cmap='gray')
plt.axis("off")
plt.title("t0",fontsize=5)

fig.text(0.01, 0.60, 'Removing noise', fontsize=5, fontweight='bold')
fig.text(0.5, 0.90, 'Time steps', fontsize=8, fontweight='bold', ha='center')
fig.text(0.01, 0.81, "Adding noise", fontsize=5, fontweight='bold')

plt.show()





