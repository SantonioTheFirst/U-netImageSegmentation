from model import build_model
import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# необходимые параметры для обучения
rs = 7
batch_size = 64
image_shape = (256, 256)
epochs = 15
verbose=True


def get_train_valid_datasets(foldername):
    '''
    Функция для создания датасетов для обучения и валидации
    '''
    train_images_ds, valid_images_ds = image_dataset_from_directory(
        f'{foldername}/image',
        labels=None,
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_shape,
        seed=rs,
        validation_split=0.02,
        subset='both'
    )

    train_masks_ds, valid_masks_ds = image_dataset_from_directory(
        f'{foldername}/mask',
        labels=None,
        class_names=None,
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_shape,
        seed=rs,
        validation_split=0.02,
        subset='both'
    )

    train_images_ds = train_images_ds.map(lambda x: x / 255.0)
    train_masks_ds = train_masks_ds.map(lambda x: x / 255.0)
    valid_images_ds = valid_images_ds.map(lambda x: x / 255.0)
    valid_masks_ds = valid_masks_ds.map(lambda x: x / 255.0)

    train_ds = tf.data.Dataset.zip((train_images_ds, train_masks_ds)).prefetch(tf.data.AUTOTUNE)
    valid_ds = tf.data.Dataset.zip((valid_images_ds, valid_masks_ds)).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds


def get_test_ds(foldername):
    '''
    Функция для создания тестового датасета
    '''
    test_images_ds = image_dataset_from_directory(
        f'{foldername}/image',
        labels=None,
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_shape,
        seed=rs
    ).map(lambda x: x / 255.0)

    test_masks_ds = image_dataset_from_directory(
        f'{foldername}/mask',
        labels=None,
        class_names=None,
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_shape,
        seed=rs
    ).map(lambda x: x / 255.0)

    test_ds = tf.data.Dataset.zip((test_images_ds, test_masks_ds)).prefetch(tf.data.AUTOTUNE)

    return test_ds


def evaluate_model(model, test_ds):
    '''
    Функция для тестирования модели и замера метрик качества
    '''
    evaluation_results = model.evaluate(test_ds)
    print(evaluation_results)
    return evaluation_results


def get_callbacks():
    '''
    Возвращает используемые колбеки
    '''
    es = EarlyStopping(patience=5, restore_best_weights=True)
    mchp = ModelCheckpoint('model.h5', save_best_only=True)
    rlr = ReduceLROnPlateau(factor=0.5, patience=1)
    return [es, mchp, rlr]


def fit_model(model, train_ds, valid_ds, batch_size, epochs, callbacks, verbose):
    '''
    Обучение
    '''
    history = model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=15,
        callbacks=callbacks,
        validation_data=valid_ds,
        verbose=verbose
    )
    return history


if __name__ == '__main__':
    m = build_model()

    print('Model built')
    train_ds, valid_ds = get_train_valid_datasets('train')
    print('Datasets loaded')
    callbacks=get_callbacks()
    print('Begin training')

    history = fit_model(
        model=m,
        train_ds=train_ds,
        valid_ds=valid_ds,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
        )

    test_ds = get_test_ds('test')
    eval = evaluate_model(m, test_ds)

    m.save_weights('model_weights.h5')