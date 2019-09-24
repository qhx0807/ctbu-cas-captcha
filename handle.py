from PIL import Image
import pytesseract

crop_box_list = [(19, 0, 32, 25), (36, 0, 49, 25), (49, 0, 62, 25)]


def convert_img(img, threshold):
    img = img.convert('L')
    pixels = img.load()
    for x in range(img.width):
        for y in range(img.height):
            if pixels[x, y] > threshold:
                pixels[x, y] = 255
            else:
                pixels[x, y] = 0
    return img


if __name__ == '__main__':
    for i in range(500):
        captcha = Image.open('./image/' + str(i) + '.png')
        res = convert_img(captcha, 150)
        for box in range(3):
            cropped = res.crop(crop_box_list[box])
            cropped.save('./cut_pic/' + str(i) + '_' + str(box) + '.png')
