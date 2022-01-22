from VideoReader import *

if __name__ == '__main__':
    print("Reading video")
    reader = VideoReader()
    reader.process_some_mp4frames('./data/ride.mp4', 5, 7, "show")