import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model



def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
    l2_reg=0.1,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
        kernel_regularizer=keras.regularizers.l2(l2_reg),  # Extra line 
    )(block_input)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DilatedSpatialPyramidPooling_mod(dspp_input):
    # # Ablation study: DSPP with huge pooling 
    # dims = dspp_input.shape
    # x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    # x = convolution_block(x, kernel_size=1, use_bias=True)
    # out_pool = layers.UpSampling2D(
    #     size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
    #     interpolation="bilinear",
    # )(x)

    # 2x2 Image pooling con reducción a 16x16
    x = layers.AveragePooling2D(pool_size=(2, 2))(dspp_input)  # Reduce de 32x32 a 16x16
    x = convolution_block(x, kernel_size=1, use_bias=True)  # Bloque convolucional de 1x1
    out_pool = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # Upsample de 16x16 a 32x32
    
    # # Ablation study: original DSPP with dilation rates 1, 6, 12, 18 
    # out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    # out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    # out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    # out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
    # x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])

    # For a 512p input, DSPP is performed over 32p area
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_2 = convolution_block(dspp_input, kernel_size=3, dilation_rate=2)
    out_4 = convolution_block(dspp_input, kernel_size=3, dilation_rate=4)
    out_8 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_2, out_4, out_8])

    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes=None, name='DeeplabV3Plus'):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output

    x = DilatedSpatialPyramidPooling(x)

    # Use fixed upsampling size
    input_a = layers.UpSampling2D(
        size=(4, 4),  # Fixed size, assuming the feature map is 1/4 of the input
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    
    # Use fixed upsampling size
    x = layers.UpSampling2D(
        size=(4, 4),  # Fixed size, assuming the feature map is 1/4 of the input
        interpolation="bilinear",
    )(x)

    if num_classes is None or num_classes == 1:
        model_output = layers.Conv2D(1, kernel_size=(1, 1), padding="same", activation='sigmoid')(x)
    else:
        model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation='softmax')(x)

    return keras.Model(inputs=model_input, outputs=model_output, name=name)

def unet(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    
    # Bottleneck
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    
    # Decoder
    up6 = UpSampling2D((2, 2))(conv5)
    up6 = Concatenate()([up6, conv4])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = UpSampling2D((2, 2))(conv6)
    up7 = Concatenate()([up7, conv3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    
    up8 = UpSampling2D((2, 2))(conv7)
    up8 = Concatenate()([up8, conv2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    
    up9 = UpSampling2D((2, 2))(conv8)
    up9 = Concatenate()([up9, conv1])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def DeeplabV3Plus_mod(image_size, num_classes=None, name='DeeplabV3Plus', backbone='xception', weights='imagenet'):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    # Ensure the input is in 0-255 range
    model_input = layers.Lambda(lambda x: x * 255)(model_input) 
    if backbone == None:
        # Normalize to 0-1 range
        model_input = layers.Lambda(lambda x: x / 255)(model_input)

        #  Block to get x (32x32x256)
        x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(model_input)  # 256x256x64
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)  # 128x128x128
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)  # 64x64x256
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)  # 32x32x256
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Block to get input_b (128x128x128)
        input_b = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(model_input)  # 256x256x64
        input_b = layers.BatchNormalization()(input_b)
        input_b = layers.ReLU()(input_b)
        
        input_b = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(input_b)  # 128x128x128
        input_b = layers.BatchNormalization()(input_b)
        input_b = layers.ReLU()(input_b)
            
    if backbone == 'mobilenetv2':
        preprocessed = keras.applications.mobilenet_v2.preprocess_input(model_input)
        base_model = keras.applications.MobileNetV2(
            weights=weights, include_top=False, input_tensor=preprocessed
        )
        x = base_model.get_layer("block_13_expand_relu").output  # Approx. 5x down
        input_b = base_model.get_layer("block_3_expand_relu").output  # Approx. 3x down

    if backbone == 'resnet101':
        preprocessed = keras.applications.resnet.preprocess_input(model_input)
        resnet101 = keras.applications.ResNet101(
            weights=weights, include_top=False, input_tensor=preprocessed
        )
        x = resnet101.get_layer("conv4_block23_2_relu").output
        input_b = resnet101.get_layer("conv2_block2_out").output

    if backbone == 'resnet50':
        preprocessed = keras.applications.resnet50.preprocess_input(model_input)
        resnet50 = keras.applications.ResNet50(
            weights=weights, include_top=False, input_tensor=preprocessed
        )
        x = resnet50.get_layer("conv4_block6_2_relu").output
        input_b = resnet50.get_layer("conv2_block3_2_relu").output

    if backbone == 'xception':
        preprocessed = keras.applications.xception.preprocess_input(model_input)
        base_model = keras.applications.Xception(
            weights=weights, include_top=False, input_tensor=preprocessed
        )
        x = base_model.get_layer("block13_sepconv2_act").output  # Approx. 5x down
        input_b = base_model.get_layer("block3_sepconv2_act").output  # Approx. 3x down
        # padding to obtain 128x128 spatial resolution bc the output is 127x127 
        input_b = tf.image.resize(input_b, (128, 128))  # layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(input_b) 
        # input_b = layers.Dropout(0.2)(input_b)  # Extra line

    x = DilatedSpatialPyramidPooling_mod(x)
    # x = convolution_block(x, num_filters=256, kernel_size=3)  # testing line 

    # Use fixed upsampling size
    input_a = layers.UpSampling2D(
        size=(4, 4),  # Fixed size, assuming the feature map is 1/4 of the input
        interpolation="bilinear",
    )(x)
    # Use transpose convolution for upsampling
    # input_a = layers.Conv2DTranspose(
    #     filters=x.shape[-1],  # Adjust the number of filters based on your model architecture
    #     kernel_size=(3, 3),  # Kernel size can be adjusted
    #     strides=(4, 4),  # Strides to double the size of the feature map
    #     padding='same',  # Use 'same' padding to keep the feature map size
    # )(x)

    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    x = layers.Concatenate(axis=-1)([input_a, input_b])

    x = convolution_block(x) 
    x = convolution_block(x)
    
    # Use fixed upsampling size
    x = layers.UpSampling2D(
        size=(4, 4),  # Fixed size, assuming the feature map is 1/4 of the input
        interpolation="bilinear",
    )(x)
    # Use transpose convolution for upsampling
    # x = layers.Conv2DTranspose(
    #     filters=x.shape[-1],  # Adjust the number of filters based on your model architecture
    #     kernel_size=(3, 3),  # Kernel size can be adjusted
    #     strides=(4, 4),  # Strides to double the size of the feature map
    #     padding='same',  # Use 'same' padding to keep the feature map size
    # )(x)

    if num_classes is None or num_classes == 1:
        model_output = layers.Conv2D(1, kernel_size=(1, 1), padding="same", activation='sigmoid')(x)
    else:
        model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation='softmax')(x)

    return keras.Model(inputs=model_input, outputs=model_output, name=name)

def DeeplabV3Plus_multichannel(image_size, num_channels=3, num_classes=None):
    model_input = keras.Input(shape=(image_size, image_size, num_channels))
    resnet50 = keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)
    # x = DilatedSpatialPyramidPooling_large(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    if num_classes is None or num_classes == 1:
        model_output = layers.Conv2D(1, kernel_size=(1, 1), padding="same", activation='sigmoid')(x)
    else:
        # model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)  # Original line
        model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation='softmax')(x)

    return keras.Model(inputs=model_input, outputs=model_output)