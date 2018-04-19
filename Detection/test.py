import tensorflow as tf
from detector import Detector



def pnet_test(prefix, epoch,batch_size,test_mode="PNet",
        thresh=[0.6, 0.6, 0.7], min_face_size=25,stride=2,
        slide_window=False, shuffle=False, vis=False):
    detectors = [None, None, None]
    print("Test model: ", test_mode)
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    print(model_path[0])
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    detectors[0] = PNet
    filename = './wider_face_train_bbx_gt.txt'
    data = read_annotation(basedir,filename)
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                            stride=stride, threshold=thresh, slide_window=slide_window)
    test_data = TestLoader(data['images'])
    detections,_ = mtcnn_detector.detect_face(test_data)

def parse_args():
        parser = argparse.ArgumentParser(description='Test mtcnn',
                                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                                            default='RNet', type=str)
                parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                                                default=['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet/ONet'],
                                                                        type=str)
                    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                                                    default=[18, 14, 22], type=int)
                        parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                                                        default=[2048, 256, 16], type=int)
                            parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                                                            default=[0.4, 0.05, 0.7], type=float)
                                parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                                                                default=24, type=int)
                                    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                                                                    default=2, type=int)
                                        parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
                                            # parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                                                #                     default=0, type=int)
                                                    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
                                                        # parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
                                                            args = parser.parse_args()
                                                                return args

if __name__ == '__main__':
        
