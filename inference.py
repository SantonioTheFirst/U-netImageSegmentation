from model import build_model
from argparse import ArgumentParser
import numpy as np
from PIL import Image


image_shape = (256, 256)


if __name__ == '__main__':
    # парсим параметры
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', nargs=1, help='Name of the image')
    parser.add_argument('-o', '--output', nargs=1, help='Name of the output image')
    args = parser.parse_args()
    input_name = str(args.input[0])
    output_name = str(args.output[0])

    # открываем и подготавливем изображение
    input_img = Image.open(input_name)
    input_shape = input_img.size
    image_np = np.asarray(input_img.resize(image_shape), dtype=np.uint8)
    preprocessed_image = np.expand_dims(image_np, axis=0) / 255.0
    # print(preprocessed_image.shape)

    # строим модель и загружаем предварительно обученные веса
    model = build_model()
    model.load_weights('weights/U-net_256_dice.h5')

    # предсказывае маску
    prediction = model.predict(preprocessed_image, verbose=False)
    # print(prediction.shape)

    # конвертируем массив в серое изображение и увеличиваем до исходных размеров, сохраняем результат
    predicted_mask = Image.fromarray(np.uint8(prediction[0] * 255).reshape(image_shape), 'L').resize(input_shape)
    predicted_mask.save(output_name)