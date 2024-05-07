import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 

import seaborn as sns

### get current work directory
current_dir = os.path.dirname(os.path.abspath(__file__))

#data_loader = Load_data(current_dir+'/HARDataset/')
print(current_dir)
# Load an image
image = cv2.imread(current_dir+'/image.jpg')





gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(gray_image)
image_mean = np.mean(gray_image)
image_variance =np.var(gray_image)
print("MEAN",image_mean)
print("variance",image_variance)

# pixel_values = gray_image.flatten()


# histogram, bins = np.histogram(pixel_values, bins=256, range=(0, 255))

# # Normalize the histogram to get probability density
# probability_density = histogram / np.sum(histogram)

# # plt.figure(figsize=(8, 6))
# # plt.bar(bins[:-1], probability_density)
# # plt.xlabel('Pixel Value')
# # plt.ylabel('Probability Density')
# # plt.title('Probability Density Distribution of the Image')
# # plt.show()

# ### frequency distribution


# # sns.displot(pixel_values, kde=True, bins=30)

# # # Set the title and axis labels
# # plt.title("Distribution of Pixel Values")
# # plt.xlabel("Pixel Value")
# # plt.ylabel("Frequency")

# # # Show the plot
# # plt.show()



# def add_gaussian_noise(image, beta):
#     """
#     Adds Gaussian noise to an image to create a noisy version with a specific mean and variance.
    
#     Args:
#         image (numpy.ndarray): The input image as a numpy array.
#         beta (float): The noise level, which determines the mean and variance of the Gaussian noise.
        
#     Returns:
#         numpy.ndarray: The noisy image as a numpy array.
#     """
#     # Calculate the mean and standard deviation of the Gaussian noise
#     mean = np.sqrt(1 - beta) * gray_image
#     std = np.sqrt(beta)
    
#     # Add Gaussian noise to the image
#     noisy_image = cv2.add(gray_image, np.random.normal(mean, std, image.shape).astype(np.uint8))
    
#     return noisy_image



# # Set the noise level (beta)
# beta = 0.2

# # Add Gaussian noise to the image
# noisy_image = add_gaussian_noise(gray_image, beta)

# noisy_image_mean = np.mean(noisy_image)
# noise_image_variance =np.var(noisy_image)
# print("noisy MEAN",noisy_image_mean)
# print("noisy variance",noise_image_variance)

# # Save the noisy image
# cv2.imwrite('noisy_image.jpg', noisy_image)


# fig=plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(gray_image,cmap='gray')
# plt.axis("off")
# plt.title("orignial")

# fig.add_subplot(1,2,2)
# plt.imshow(noisy_image,cmap='gray')
# plt.axis("off")
# plt.title("noisy")
# plt.show()

z= np.sqrt(0.8)*114
print(z)
image_mean = np.mean(gray_image)
image_variance =np.var(gray_image)
print("MEAN",image_mean)
print("variance",image_variance)

z= np.sqrt(0.8)*gray_image
image_mean_z =np.mean(z)
image_variance_z =np.var(z)
print("MEAN_z",image_mean_z)
print("variance_z",image_variance_z)

def conditional_probability(image, mean, variance):
    # Calculate the conditional probability distribution
    prob = np.exp(-0.5 * ((image - mean) ** 2) / variance) / np.sqrt(2 * np.pi * variance)
    return prob



# Normalize the histogram to get probability density

