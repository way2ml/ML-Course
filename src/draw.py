import numpy as np
from PIL import Image
import cv2

Chanel = 1
PaintSize = 280
BgColor = 0
PaintColor = (155)
StrokeWeight = 20

drawing = False
start = (-1, -1)
lastPoint = (-1,-1)


# 初始化画板背景色为白色
img = np.full((PaintSize, PaintSize,Chanel), BgColor, dtype=np.uint8)

# 有鼠标事件就会调用下面这个函数画图
def mouse_event(event, x, y, flags, param):
    global drawing, img, lastPoint
    # 如果鼠标左键被按下
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        lastPoint = (x, y)
        start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img,lastPoint,(x, y), PaintColor, StrokeWeight)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    
    # 更新上一个点
    lastPoint = (x, y)


# 图像预处理
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28,28), True) # 将1D存储的数据改变成2D
    im_arr = np.array(reIm.convert('L'))  # 将PIL.Image.Image格式转化为numpy.ndarray方便显示
    cv2.imshow("Little",im_arr) # 显示缩小后的图像
    
    # 改变数据的形状便于喂入神经网络
    nm_arr = im_arr.reshape([1,784])
    nm_arr = nm_arr.astype(np.float32)
    # 将0～255映射到0到1
    img_ready = np.multiply(nm_arr, 1.0/255.0)
    return img_ready



# 应用程序
def application():
    global img
    cv2.namedWindow('Press \'s\' to Save,\'c\' to Clear')
    cv2.setMouseCallback('Press \'s\' to Save,\'c\' to Clear', mouse_event)
    print("Press q or Esc to quit the program:")
    while True:
        cv2.imshow('Press \'s\' to Save,\'c\' to Clear', img)
        key = cv2.waitKey(20)
        if key == 27 or key == 113: # 按`q`或者`Esc`退出
            break
        elif key == 115: # 按`s`保存图片
            data = img.flatten()
            data = data.reshape((PaintSize, PaintSize))
            data = data[::10,::10]/255.
            np.save('test_img.npy', data.reshape((-1,1)))
            print('Image saved.')
        elif key == 99: # 按`c`清空画图板
            img = np.full((PaintSize, PaintSize,Chanel), BgColor, dtype=np.uint8)
        else:
            pass

def main():
    application()
    
if __name__ == '__main__':
    main()

# %%
