import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import os

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


# 1. 重新构建与训练时完全一致的模型
def build_model(class_num=325, IMG_SHAPE=(224, 224, 3)):
    # 加载预训练的MobileNetV2模型
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # 保持与训练时一致

    # 构建与训练时完全相同的模型结构
    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])

    return model


# 2. 数据加载函数
def data_load(data_dir, valid_data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=100,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_data_dir,
        label_mode='categorical',
        seed=100,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


# 3. 测试函数
def test_mobilenet():
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)

    # 加载数据
    train_ds, test_ds, class_names = data_load(
        "train",
        "valid",
        224, 224,
        12
    )
    class_num = len(class_names)

    # 构建新模型（与训练时结构完全一致）
    model = build_model(class_num=class_num)

    # 加载权重（只加载权重，不加载模型结构）
    try:
        model.load_weights("models/mobilenet_fv_20241218142756.h5")
        print("权重加载成功")
    except Exception as e:
        print(f"权重加载失败: {str(e)}")
        # 尝试兼容模式加载
        try:
            model.load_weights("models/mobilenet_fv_20241218142756.h5", by_name=True)
            print("按名称匹配加载权重成功")
        except Exception as e2:
            print(f"按名称匹配加载失败: {str(e2)}")
            return

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 模型评估
    loss, accuracy = model.evaluate(test_ds)
    print(f'Mobilenet test accuracy : {accuracy}')

    # 收集真实标签和预测标签
    test_real_labels = []
    test_pre_labels = []
    for test_batch_images, test_batch_labels in test_ds:
        test_batch_labels = test_batch_labels.numpy()
        test_batch_pres = model.predict(test_batch_images, verbose=0)

        test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
        test_batch_pres_max = np.argmax(test_batch_pres, axis=1)

        test_real_labels.extend(test_batch_labels_max)
        test_pre_labels.extend(test_batch_pres_max)

    # 生成混淆矩阵
    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for real, pred in zip(test_real_labels, test_pre_labels):
        heat_maps[real][pred] += 1

    # 归一化处理
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    heat_maps_sum[heat_maps_sum == 0] = 1  # 避免除零错误
    heat_maps_float = heat_maps / heat_maps_sum

    # 创建并保存交互式热力图
    fig = px.imshow(
        heat_maps_float,
        x=class_names,
        y=class_names,
        labels=dict(x="预测标签", y="实际标签", color="比例"),
        title="混淆矩阵热力图"
    )
    fig.write_html("results/heatmap_mobilenet_interactive.html")
    fig.show()


if __name__ == '__main__':
    test_mobilenet()
