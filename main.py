from WaveDec import *
from VideoReader import *

if __name__ == '__main__':
    print("Reading video")
    # reader = VideoReader()
    # reader.set_instream('./data/ride.mp4')
    # reader.process_all_mp4frames('./data/ride.mp4', "write_mp4", "./data/ride_with_cv2.mp4")

    wc = WaveDec('./data/ride.mp4', "./data/why.mp4", 'db2', 3)
    #wc.test_independent_frame_compression()
    wc.test_decompression()

