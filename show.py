from PIL import Image
import argparse
import numpy as np

import tensorflow as tf

from cam import EigenCAM





def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='tensorflow model path (keras saved model folder)')
    parser.add_argument('-d', default='cpu', help='device ("cpu" or "cuda")')
    parser.add_argument('-img', help='img path')
    parser.add_argument('-layer', help='layer name to plot heatmap')
    return parser.parse_args()
    



def main():
    arg = get_arg()

    preprocess = tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224),
        tf.keras.layers.Rescaling(1 / 127.5, -1),
    ])
   
    device = arg.d
    if device == 'cuda' and len(tf.config.list_physical_devices('GPU')) == 0:
        raise ValueError('There is no cuda !!!')
    
    if arg.m is None:
        model = tf.keras.applications.MobileNetV2(classifier_activation=None)
    else:
        model = tf.keras.models.load_model(arg.m)
    model.summary()
        
    cam_obj = EigenCAM(model, arg.d, preprocess, arg.layer)
    
    print('\ndevice:', arg.d)
    print('layer Name to plot heatmap:', cam_obj.layer_name)
    print('img:', arg.img)
    
    img = np.array(Image.open(arg.img).convert('RGB'))
    # output is tf Tensor, overlay is ndarray
    output, overlay = cam_obj.get_heatmap(img)
    print('\nPredict label:', np.argmax(output, 1).item())
    
    Image.fromarray(overlay).show()





if __name__ == "__main__":
    main()
