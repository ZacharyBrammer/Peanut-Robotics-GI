from PIL import Image

def rotate_image(room_image):
    return room_image.resize((256, 256)).rotate(-90).resize((192, 256))
