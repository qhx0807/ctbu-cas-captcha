# ctbu-cas-captcha

学校统一身份认证平台登录验证码识别

![](./0.png)

## usage

```python
def check_code():
    clf = joblib.load('train_model.m')
    image = Image.open('268.png')
    res = convert_img(image, 150)
    for i in range(3):
        cropped = res.crop(crop_box_list[i])
        feat_list = get_feature(cropped)
        y_pred = clf.predict([feat_list])
        print(y_pred)
```