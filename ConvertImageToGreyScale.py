# example of saving a grayscale version of a loaded image
from PIL import Image
# load the image
image = Image.open('Sydney-Opera-House.jpeg')
# convert the image to grayscale
gs_image = image.convert(mode='L')
# save in jpeg format
gs_image.save('Sydney-Opera-House.jpeg')
# load the image again and show it
image2 = Image.open('Sydney-Opera-House.jpeg')
# show the image
image2.show()
