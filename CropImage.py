# example of cropping an image
from PIL import Image
# load image
image = Image.open('Sydney-Opera-House.jpeg')
# create a cropped image
cropped = image.crop((100, 100, 200, 200))
# show cropped image
cropped.show()
