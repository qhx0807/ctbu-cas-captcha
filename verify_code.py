import os
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

crop_box_list = [(19, 0, 32, 25), (36, 0, 49, 25), (49, 0, 62, 25)]


def get_feature(img):
    """
    获取指定图片的特征值,
    一张截图为高25，宽13。统计各行的黑点，得到25个信息，统计各列的黑点，得到13个信息，用列表返回38个信息
    """
    pixel_cnt_list = []
    height = 25
    width = 13
    for y in range(height):
        pix_cnt_x = 0
        for x in range(width):
            if img.getpixel((x, y)) == 0:
                pix_cnt_x += 1
        pixel_cnt_list.append(pix_cnt_x)

    for x in range(width):
        pix_cnt_y = 0
        for y in range(height):
            if img.getpixel((x, y)) == 0:
                pix_cnt_y += 1
        pixel_cnt_list.append(pix_cnt_y)
    return pixel_cnt_list


def sort_table():
    sort_list = []
    files_list = os.listdir('./cut_pic')
    num = 0
    for i in files_list:
        files = os.listdir('./cut_pic/' + i)
        for j in range(0, len(files)):
            image = Image.open('./cut_pic/' + i + '/' + files[j])
            sort_list.append(get_feature(image))
            sort_list[num].append(i)
            num += 1
    return sort_list


def save_text():
    names = [x for x in range(1, 40)]
    sort_list = sort_table()
    test = pd.DataFrame(sort_list, columns=names)
    test.to_csv('data.txt')


def train():
    data = pd.read_csv('data.txt')
    clf = OneVsOneClassifier(SVC(kernel='linear'))

    x = data.iloc[:, 1:39]
    y = np.array(data.iloc[:, 39]).astype(str)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    rf = pd.DataFrame(list(zip(y_pred, y_test)), columns=['predicted', 'actual'])  # rf类型是dataframe
    rf['correct'] = rf.apply(lambda r: 1 if r['predicted'] == r['actual'] else 0, axis=1)
    print(len(rf[rf['correct'] == 1]) / len(rf))
    joblib.dump(clf, "train_model.m")
    print('训练完毕')


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


def check_code():
    clf = joblib.load('train_model.m')
    image = Image.open('268.png')
    res = convert_img(image, 150)
    for i in range(3):
        cropped = res.crop(crop_box_list[i])
        feat_list = get_feature(cropped)
        y_pred = clf.predict([feat_list])
        print(y_pred)


if __name__ == '__main__':
    check_code()
