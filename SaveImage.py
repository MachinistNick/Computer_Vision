# example of saving an image in another format
from PIL import Image
# load the image
image = Image.open('Sydney-Opera-House.jpeg')
# save as PNG format
image.save('Sydney-Opera-House.png', format='PNG')
# load the image again and inspect the format
image2 = Image.open('Sydney-Opera-House.png')
print(image2.format)
