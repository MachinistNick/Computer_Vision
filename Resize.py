# resize image and force a new shape
from PIL import Image
# load the image
image = Image.open('Sydney-Opera-House.jpeg')
# report the size of the image
print(image.size)
# resize image and ignore original aspect ratio
img_resized = image.resize((200,200))
# report the size of the thumbnail
print(img_resized.size)
