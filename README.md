# Photo-Restoration

# Team Members
C22305863 - Iria Parada
C22308773 - Amy Ibourk
C22359751 - Arushi Singh

# Introduction
Restoring photos from the past! Making sure they are clearer, sharper and rid of blemishes. 

# Contributions - Iria Parada
For my part of the project, I focused on creating a system that first checks whether an image actually needs certain operations before applying them. I started by detecting whether the photo was noisy enough to require Gaussian blur, using the Laplacian of the grayscale image and calculating its variance to estimate noise levels. [1]
```
def needs_gaussian(photo, threshold=300): 
    #convert to greyscale
    grey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY) 
    #apply laplacian  to highlight edges and noise 
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    #calcuate variance of the laplacian  
    noise_level = lap.var() 

    #if noise is abouve threshold return true 
    return noise_level > threshold
```
The result is then compared with a threshold and checked if it needs the Gaussian filter or not. The gaussian() function I created applies the cv2 GaussianBlur. “Gaussian filter is used for blurry pictures and to take away noise. The amount of smoothing is decided by the quality eccentricity of the Gaussian” [2].
```
def gaussian(photo, ksize=3, sigma=0.5): 
    return cv2.GaussianBlur(photo, (ksize, ksize), sigma) 
```
For the inpainting, I created a needs_inpaint() function. Similar to the Gaussian check, I use the grayscale version again to create a mask of pixels with brightness between 110 and 160, and then calculate the damage ratio of the mask by counting the white pixels.
```
def needs_inpaint(photo):
    #convert to greyscale 
    grey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY) 
    #create mask of pixels with brightness in between 110 and 160
    mask_mid = cv2.inRange(grey, 110, 160) 
 
    #calculate if mask contains damage
    damage_ratio = np.count_nonzero(mask_mid) / mask_mid.size
     
    #if more than 1% damage return true 
    return damage_ratio > 0.01  
``` 
The inpainting function also uses the grayscale image and selects the mid-intensity pixels where scratches would appear. Since this raw mask picks up noise too, I applied several morphological filtering steps; opening and closing with different kernel sizes, to remove small dots, refine the shapes, and avoid false positives. Once the mask was cleaned, I used cv2.threshold() to convert it into a proper binary mask for inpainting. The inpainting itself was done using the cv2 inpaint method, which fills the damaged regions by smoothly interpolating from the surrounding pixels [3]. This worked well for long hairline scratches and small patches of wear.
```
def inpainting(photo):
    #convert to greyscale
    grey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    #select mid-intensity pixels where scratches appear
    mask_mid = cv2.inRange(grey, 110, 160) 
    #create a kernel for morphological filtering
    kernel = np.ones((5,5), np.uint8)
     
    #opening to remove small isolated noise 
    mask_mid = cv2.morphologyEx(mask_mid, cv2.MORPH_OPEN, kernel)
    #closing to fill tiny hiles detected 
    mask_mid = cv2.morphologyEx(mask_mid, cv2.MORPH_CLOSE, kernel)
 
    #another opening to remove bigger spots(reduce false positives)
    mask_small = cv2.morphologyEx(mask_mid, cv2.MORPH_OPEN, np.ones((9,9), np.uint8))
    #conevet into binary mask
    _, damage_mask = cv2.threshold(mask_small, 127, 255, cv2.THRESH_BINARY)
 
    #use inpaint to fill in the damaged areas 
    restored = cv2.inpaint(photo, damage_mask, 5, cv2.INPAINT_TELEA)
 
    return damage_mask, restored
```

# Contributions - Amy Ibourk

# Contributions - Arushi Singh

# Conclusion

# References
[1]

Sagar, “Laplacian and its use in Blur Detection,” Medium, May 22, 2020. https://medium.com/@sagardhungel/laplacian-and-its-use-in-blur-detection-fbac689f0f88

‌[2]

Priyanka, R. Shukla, and L. Shrivastava, “Image Restoration of Image with Gaussian Filter.” Accessed: Nov. 29, 2025. [Online]. Available: https://www.irjet.net/archives/V7/i12/IRJET-V7I1299.pdf

[3]

A. Sharma, “How to Repair Damaged Images using Inpainting methods in Python using OpenCV,” Medium, Feb. 07, 2022. https://towardsdev.com/how-to-repair-damaged-images-using-inpainting-methods-in-python-using-opencv-339982754137 (accessed Nov. 29, 2025).

‌[4]

“The AI workspace that works for you. | Notion,” Notion, 2025. https://www.notion.so/BLOGS-2b705bf5be1c80bda8c4f380bb081275?source=copy_link#2ba05bf5be1c8019a237cb324b736acd (accessed Nov. 29, 2025).

‌[5]

atduskgreg, “opencv-processing-book/book/filters/in_range.md at master · atduskgreg/opencv-processing-book,” GitHub, 2025. https://github.com/atduskgreg/opencv-processing-book/blob/master/book/filters/in_range.md

[6]

GeeksforGeeks, “Simple Thresholding using OpenCV,” GeeksforGeeks, May 06, 2019. https://www.geeksforgeeks.org/python/simple-thresholdin-using-opencv/

[7]

R. Singh, “Morphological Transformations of Images using OpenCV | Image Processing Part-2,” Analytics Vidhya, Nov. 05, 2020. https://medium.com/analytics-vidhya/morphological-transformations-of-images-using-opencv-image-processing-part-2-f64b14af2a38

‌[8]

https://facebook.com/coseriescommunity and https://facebook.com/coseriescommunity, “Morphological Transformations in Python using OpenCV,” coseries, Nov. 22, 2020. https://coseries.com/morphological-transformations-in-python-using-opencv/ (accessed Nov. 29, 2025).

[9]

P. Team, “Morphological Operations in OpenCV,” Python Geeks, Feb. 14, 2023. https://pythongeeks.org/morphological-operations-in-opencv/ (accessed Nov. 29, 2025).

‌[10]

“OpenCV Smoothing and Blurring,” PyImageSearch, Apr. 28, 2021. https://pyimagesearch.com/2021/04/28/opencv-smoothing-and-blurring/

[11]

A. Rosebrock, “Image inpainting with OpenCV and Python,” PyImageSearch, May 18, 2020. https://pyimagesearch.com/2020/05/18/image-inpainting-with-opencv-and-python/

‌

