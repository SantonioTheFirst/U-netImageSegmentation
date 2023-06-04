from model import build_model
import numpy as np
from PIL import Image
import streamlit as st


image_shape = (256, 256)
# строим модель и загружаем предварительно обученные веса
model = build_model()
model.load_weights('weights/U-net_256_dice.h5')


st.set_page_config(page_title='U-net image segmentation example', page_icon=':ship:')

'''
# Hello!

This toy website allows you to interact with [U-net segmentation model](https://github.com/SantonioTheFirst/U-netImageSegmentation) trained to segment images with ships.

Just upload your image with ships below!
'''

file = st.file_uploader('Upload your ships', accept_multiple_files=False)

try:
    if file:
        st.image(file)
        with Image.open(file) as im:
            input_shape = im.size
            image_np = np.asarray(im.resize(image_shape), dtype=np.uint8)
            preprocessed_image = np.expand_dims(image_np, axis=0) / 255.0

            # предсказывае маску
            prediction = model.predict(preprocessed_image, verbose=False)

            preprocessed_image_cropped_by_mask = preprocessed_image * prediction

            # print(preprocessed_image_cropped_by_mask.shape)

            # # конвертируем массив в серое изображение и увеличиваем до исходных размеров, сохраняем результат
            predicted_mask = Image.fromarray(np.uint8(prediction[0] * 255).reshape(image_shape), 'L').resize(input_shape)
            
            st.image(predicted_mask)


            preprocessed_image_cropped_by_mask = Image.fromarray(
                np.uint8(preprocessed_image_cropped_by_mask[0] * 255)
            ).resize(input_shape)

            st.image(preprocessed_image_cropped_by_mask)

except Exception:
    '''
    Oops, something is wrong!
    '''