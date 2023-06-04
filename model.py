from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout, Lambda, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# все необходимые функции

def dice_coef(y_true, y_pred, smooth=100):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def build_model(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
    input = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255.0)(input)
    c1 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(input)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p2)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p3)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p4)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c5)

    # up
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5) #c5
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7) #c7
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c9)

    output = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input], outputs=[output])
    # model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=dice_coef_loss, metrics=['accuracy', dice_coef])
    return model


if __name__ == '__main__':
    m = build_model()
    print(type(m))