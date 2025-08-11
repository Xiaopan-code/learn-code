# -*- coding: utf-8 -*-
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil
import os


# 构建模型，支持手动指定权重文件
def build_compatible_model(num_classes, weights_path=None):
    # 加载MobileNet，不自动下载权重
    base_model = MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # 不自动下载权重
    )

    # 如果提供了权重文件路径，加载权重
    if weights_path and os.path.exists(weights_path):
        base_model.load_weights(weights_path)
        print(f"成功加载本地权重文件: {weights_path}")
    else:
        print("未加载预训练权重，使用随机初始化权重")

    # 冻结基础模型
    base_model.trainable = False

    # 添加自定义分类层
    input_tensor = Input(shape=(224, 224, 3))
    x = base_model(input_tensor, training=False)
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=input_tensor, outputs=output_tensor)


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/logo.png'))
        self.setWindowTitle('鸟类识别系统')

        # 鸟类类别列表
        self.class_names = [' 非洲冕鹤 ', ' 非洲火雀 ', ' 信天翁 ', ' 亚历山大鹦鹉 ', ' 美洲反嘴鹬 ', ' 美洲麻鳽 ',
                            ' 美洲白骨顶 ', ' 美洲金翅雀 ', ' 美洲红隼 ', ' 美洲鹨 ', ' 美洲红尾鸲 ', ' 蛇鹈 ',
                            ' 安娜蜂鸟 ', ' 蚁鸟 ', ' 阿拉里皮娇鹟 ', ' 朱鹮 ', ' 白头海雕 ', ' 秃鹮 ', ' 巴厘岛八哥 ',
                            ' 巴尔的摩黄鹂 ', ' 蕉森莺 ', ' 斑阔嘴鸟 ', ' 斑八色鸫 ', ' 斑尾塍鹬 ', ' 仓鸮 ', ' 家燕 ',
                            ' 横斑蓬头䴕 ', ' 栗胸林莺 ', ' 须䴕 ', ' 须钟雀 ', ' 须苇莺 ', ' 斑鱼狗 ', ' 极乐鸟 ',
                            ' 黑黄阔嘴鸟 ', ' 黑冠鹃隼 ', ' 黑鹧鸪 ', ' 黑剪嘴鸥 ', ' 黑天鹅 ', ' 黑尾苦恶鸟 ',
                            ' 黑喉山雀 ', ' 黑喉林莺 ', ' 黑兀鹫 ', ' 黑顶山雀 ', ' 黑颈䴙䴘 ', ' 黑喉麻雀 ',
                            ' 黑伯劳林莺 ', ' 金冠啄木鸟 ', ' 蓝美洲鹫 ', ' 蓝松鸡 ', ' 蓝鹭 ', ' 蓝喉小巨嘴鸟 ',
                            ' 食米鸟 ', ' 婆罗洲须鴷 ', ' 婆罗洲叶鹎 ', ' 婆罗洲雉鸡 ', ' 布兰特鸬鹚 ', ' 褐顶雀鹛 ',
                            ' 褐噪鸫 ', ' 褐弯嘴嘲鸫 ', ' 白腹锦鸡 ', ' 仙人掌鹪鹩 ', ' 加利福尼亚兀鹫 ',
                            ' 加利福尼亚鸥 ', ' 加利福尼亚鹌鹑 ', ' 金丝雀 ', ' 海角辉椋鸟 ', ' 黑顶白颊林莺 ',
                            ' 冕苍鹭 ', ' 伞鸟 ', ' 红喉蜂虎 ', ' 里海鸬鹚 ', ' 食火鸡 ', ' 雪松太平鸟 ',
                            ' 蓝翅黄森莺 ', ' 领环颈鹦鹉 ', ' 栗腹歌鹀 ', ' 雀鹀 ', ' 凤头海雀 ', ' 凤头卡拉鹰 ',
                            ' 凤头山雀 ', ' 朱红蜂鸟 ', ' 乌鸦 ', ' 凤头鸠 ', ' 古巴咬鹃 ', ' 古巴翠鸟 ',
                            ' 卷冠阿拉卡䴕 ', ' 达氏须䴕 ', ' 暗眼灯草鹀 ', ' 双斑雀 ', ' 双冠鸬鹚 ', ' 绒啄木鸟 ',
                            ' 东蓝鸲 ', ' 东草地鹨 ', ' 东玫瑰鹦鹉 ', ' 东林莺 ', ' 丽色凤头鹃 ', ' 白颈长尾雉 ',
                            ' 帝企鹅 ', ' 鸸鹋 ', ' 恩加诺岛八哥 ', ' 欧亚金黄鹂 ', ' 欧亚喜鹊 ', ' 黄昏蜡嘴雀 ',
                            ' 和平鸟 ', ' 火尾绿鹛 ', ' 火焰唐纳雀 ', ' 火烈鸟 ', ' 军舰鸟 ', ' 甘贝尔鹌鹑 ',
                            ' 刚冈凤头鹦鹉 ', ' 吉拉啄木鸟 ', ' 金翅啄木鸟 ', ' 彩鹮 ', ' 灰伯劳 ', ' 灰王霸鹟 ',
                            ' 灰山鹑 ', ' 大灰鸮 ', ' 大食蝇霸鹟 ', ' 大林鸱 ', ' 大艾草松鸡 ', ' 绿阔嘴鸟 ',
                            ' 绿松鸦 ', ' 绿鹊 ', ' 灰斑鸻 ', ' 沟嘴犀鹃 ', ' 蕉鹃 ', ' 珠鸡 ', ' 矛隼 ', ' 丑鸭 ',
                            ' 角雕 ', ' 夏威夷雁 ', ' 盔鵙 ', ' 绿尾虹雉 ', ' 麝雉 ', ' 凤头潜鸭 ', ' 戴胜 ', ' 犀鸟 ',
                            ' 角冠雉 ', ' 角百灵 ', ' 角蜂鸟 ', ' 家朱雀 ', ' 家麻雀 ', ' 紫蓝金刚鹦鹉 ', ' 帝企鹅 ',
                            ' 印加燕鸥 ', ' 印度鸨 ', ' 印度八色鸫 ', ' 印度佛法僧 ', ' 靛蓝彩鹀 ', ' 伊维鸟 ',
                            ' 裸颈鹳 ', ' 爪哇禾雀 ', ' 鹭鹤 ', ' 鸮鹦鹉 ', '小水鸟', ' 王鹫 ', ' 几维鸟 ', ' 笑翠鸟 ',
                            ' 百灵鹀 ', ' 琉璃彩鹀 ', ' 丁香色佛法僧 ', ' 长耳鸮 ', ' 鹊鹅 ', ' 马来犀鸟 ', ' 翠鸟 ',
                            ' 马达加斯加绣眼鸟 ', ' 马累冢雉 ', ' 绿头鸭 ', ' 鸳鸯 ', ' 红树林杜鹃 ', ' 秃鹳 ',
                            ' 蓝脸鲣鸟 ', ' 白颈麦鸡 ', ' 绿尾雉 ', ' 哀鸽 ', ' 八哥 ', ' 尼科巴鸠 ', ' 嘈杂钟雀 ',
                            ' 北美红雀 ', ' 北扑翅䴕 ', ' 北暴风鹱 ', ' 北鲣鸟 ', ' 苍鹰 ', ' 北水雉 ', ' 北嘲鸫 ',
                            ' 北森莺 ', ' 北红主教鸟 ', ' 琵嘴鸭 ', ' 眼斑火鸡 ', ' 冲绳秧鸡 ', ' 橙胸彩鹀 ',
                            ' 东方角鸮 ', ' 鹗 ', ' 鸵鸟 ', ' 灶鸟 ', ' 蛎鹬 ', ' 彩鹀 ', ' 帕里拉雀 ', ' 极乐唐纳雀 ',
                            ' 凤头海雀 ', ' 大山雀 ', ' 巴塔哥尼亚岭雀 ', ' 孔雀 ', ' 鹈鹕 ', ' 游隼 ', ' 菲律宾鹰 ',
                            ' 粉红知更鸟 ', ' 中贼鸥 ', ' 海鹦 ', ' 紫朱雀 ', ' 紫青水鸡 ', ' 紫崖燕 ', ' 紫水鸡 ',
                            ' 侏翠鸟 ', ' 格查尔鸟', ' 彩虹吸蜜鹦鹉 ', ' 刀嘴海雀 ', ' 红须蜂虎 ', ' 红腹八色鸫 ',
                            ' 红眉金翅雀 ', ' 红脸鸬鹚 ', ' 红脸林莺 ', ' 红梅花雀 ', ' 红头潜鸭 ', ' 红头啄木鸟 ',
                            ' 红蜜旋木雀 ', ' 红顶咬鹃 ', ' 红尾鹰 ', ' 红尾鸫 ', ' 红翅黑鹂 ', ' 红须鹎 ',
                            ' 摄政园丁鸟 ', ' 环颈雉 ', ' 走鹃 ', ' 知更鸟 ', ' 原鸽 ', ' 玫瑰脸情侣鹦鹉 ', ' 棕尾鵟 ',
                            ' 皇家姬鹟 ', ' 红玉喉蜂鸟 ', ' 红腹翠鸟 ', ' 棕腹翠鸟 ', ' 棕尾拟啄木鸟 ', ' 三宝鸟 ',
                            ' 黑枕王鹟 ', ' 沙丘鹤 ', ' 红腹角雉 ', ' 猩红顶果鸠 ', ' 猩红朱鹭 ', ' 绯红金刚鹦鹉 ',
                            ' 猩红唐纳雀 ', ' 鲸头鹳 ', ' 短嘴半蹼鹬 ', ' 史密斯长刺歌雀 ', ' 雪鹭 ', ' 雪鸮 ',
                            ' 黑脸田鸡 ', ' 亮丝鹟 ', ' 华丽鹪鹩 ', ' 勺嘴鹬 ', ' 琵鹭 ', ' 斑猫鸟 ', ' 斯里兰卡蓝鹊 ',
                            ' 船鸭 ', ' 鹳嘴翠鸟 ', ' 草莓雀 ', ' 条纹猫头鹰 ', ' 条纹娇鹟 ', ' 条纹燕 ', ' 超级椋鸟 ',
                            ' 白腹锦鸡 ', ' 台湾蓝鹊 ', ' 南秧鸡 ', ' 塔斯马尼亚鸨 ', ' 绿翅鸭 ', ' 山雀 ', ' 巨嘴鸟 ',
                            ' 汤森氏林莺 ', ' 树燕 ', ' 热带王霸鹟 ', ' 号手天鹅 ', ' 土耳其秃鹫 ', ' 绿咬鹃 ',
                            ' 伞鸟 ', ' 杂色鸫 ', ' 委内瑞拉拟黄鹂 ', ' 朱红姬鹟 ', ' 维多利亚凤冠鸠 ', ' 绿紫燕 ',
                            ' 鹫珠鸡 ', ' 攀雀 ', ' 肉垂凤冠雉 ', ' 白腰杓鹬 ', ' 白眉苦恶鸟 ', ' 白颊蕉鹃 ',
                            ' 白颈渡鸦 ', ' 白尾热带鸟 ', ' 白喉蜂虎 ', ' 野火鸡 ', ' 威尔逊极乐鸟 ', ' 林鸳鸯 ',
                            ' 黄腹啄花鸟 ', ' 黄鹎 ', ' 黄头黑鹂 ']
        num_classes = len(self.class_names)

        # 手动指定权重文件路径（请将下载的权重文件放在此路径）
        weights_path = "models/mobilenet_1_0_224_tf_no_top.h5"

        # 构建模型
        self.model = build_compatible_model(num_classes, weights_path)

        # 尝试加载原有模型权重（可选）
        try:
            self.model.load_weights("models/mobilenet_fv.h5", by_name=True)
            print("成功加载部分原有模型权重")
        except:
            print("未加载原有模型权重，使用基础模型运行")

        self.to_predict_name = "images/tim9.jpeg"
        self.resize(900, 700)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("样本")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        if img_init is None:
            # 如果初始图片不存在，使用空白图片
            img_init = np.ones((400, 400, 3), dtype=np.uint8) * 240
            cv2.putText(img_init, "无初始图片", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传图片 ")
        btn_change.setFont(font)
        btn_change.clicked.connect(self.change_img)
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)
        label_result = QLabel(' 鸟类名称 ')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)

        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用鸟类识别系统')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/bj.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel("作者：可爱小天施")
        label_super.setFont(QFont('楷体', 12))
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        self.addTab(main_widget, '主页')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image files(*.jpg *.png *.jpeg)')
        img_name = openfile_name[0]
        if img_name == '':
            return

        try:
            target_image_name = "images/tmp_up." + img_name.split(".")[-1]
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            cv2.imwrite("images/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.png"))
            self.result.setText("等待识别")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"处理图片时出错：{str(e)}")

    def predict_img(self):
        try:
            img = Image.open('images/target.png')
            img = np.asarray(img)
            if len(img.shape) == 2:  # 处理灰度图
                img = np.stack((img,) * 3, axis=-1)
            img = img / 255.0  # 归一化
            outputs = self.model.predict(img.reshape(1, 224, 224, 3))
            result_index = int(np.argmax(outputs))
            result = self.class_names[result_index]
            self.result.setText(result)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"识别时出错：{str(e)}")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出确认', '确定要退出程序吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    # 创建必要的目录
    if not os.path.exists("images"):
        os.makedirs("images")
    if not os.path.exists("models"):
        os.makedirs("models")

    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
