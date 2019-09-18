import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from utils.visualization_utils import visualize_boxes_and_labels_on_image_array
from utils.mscoco_label_map import get_mscoco_label_map



def load_image(image_path, show=False):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    (im_width, im_height) = image.size
    image = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    if show:
        plt.imshow(image)
        plt.show()
    return image


test_image_path = 'images/image1.jpg'
test_image = load_image(test_image_path, show=False)
# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
test_image_expanded = np.expand_dims(test_image, axis=0)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'pretrained_models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    with tf.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: test_image_expanded})
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        print()
        # Visualization of the results of a detection.
        category_index = get_mscoco_label_map('images/mscoco_label_map.txt')
        visualize_boxes_and_labels_on_image_array(
            test_image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        plt.figure()
        plt.imshow(test_image)
        plt.show()
