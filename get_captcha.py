import requests
import base64
url = 'https://cas.ctbu.edu.cn/lyuapServer/kaptcha?_t=1568624902&uid=50b27cd6453d454196299d29a49f5d67'


def get_imgs(num):
    """ 获取图片并保存 """
    res = requests.get(url).json()
    bs64str = res['content'].split(',')[-1].replace('%0A', '')
    byte_data = base64.b64decode(bs64str)
    name = './image/{}.png'.format(num)
    with open(name, 'wb') as f:
        f.write(byte_data)
        f.close()


if __name__ == '__main__':
    for i in range(500):
        get_imgs(i)
        print(i)
