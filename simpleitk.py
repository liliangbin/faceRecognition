import SimpleITK as sitk
import numpy as np
import pydicom as dicom
from matplotlib import pyplot as plt


def to_hu(image):
    MIN_BLOOD = -1000
    MAX_BLOOD = 400
    image = (image - MIN_BLOOD) / (MAX_BLOOD - MIN_BLOOD)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

if __name__ == '__main__':
    image = sitk.ReadImage('20016.dcm')
    image_array = sitk.GetArrayFromImage(image)
    image_array[image_array == -2000] = 0
    image_array = to_hu(image_array)

    img = dicom.read_file('20016.dcm')
    print(image_array)
    print(img.dir())
    print(img.RescaleSlope)
    print(img.RescaleIntercept)
    HU = np.dot(image_array, img.RescaleSlope) + img.RescaleIntercept
    print(HU.shape)
    print(HU)
    images = np.squeeze(HU)

    plt.imshow(images, cmap="gray")
    plt.axis("off")
    plt.savefig( "new.png")
    print("image_array done")
    plt.show()
