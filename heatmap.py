#import libraries
import numpy as np
import cv2
import copy
from progress.bar import Bar
import streamlit as st
from PIL import Image
import tempfile


def load_video(video):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())
    capture = cv2.VideoCapture(tfile.name)
    return capture


def heatmap(capture, quality):
    background_substractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = Bar('Processing Frames', max=length)
    first_iteration_indicator = 1
    for i in range(0, length-1):
        ret, frame = capture.read()
        
        scale_percent = quality # percent of original size
        width = int(frame.shape[:2][1] * scale_percent / 100)
        height = int(frame.shape[:2][0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        if first_iteration_indicator == 1:
            first_frame = copy.deepcopy(resized)
            height, width = resized.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:
            filter = background_substractor.apply(resized)
            cv2.imwrite('./frame.jpg', resized)
            cv2.imwrite('./diff_background_frame.jpg', filter)

            threshold = 2
            maxValue = 2
            ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)

            accum_image = cv2.add(accum_image, th1)
            cv2.imwrite('./mask.jpg', accum_image)
            
            color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_SUMMER)

            video_frame = cv2.addWeighted(resized, 1, color_image_video, 0.5, 0)

            name = "./frames/frame%d.jpg" %i
            cv2.imwrite(name, video_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        bar.next()
    bar.finish()


    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    width_original = int(result_overlay.shape[:2][1] / scale_percent * 100)
    height_original = int(result_overlay.shape[:2][0] / scale_percent * 100)
    dim_original = (width_original, height_original)

    # resize image
    result_resized = cv2.resize(result_overlay, dim_original, interpolation = cv2.INTER_AREA)



    capture.release()
    cv2.destroyAllWindows()
    return result_resized


def main():
    """Heatmap App"""
    st.title("Heatmap")

    video_file = st.file_uploader("Upload Video", type=['mp4'])
    my_range = range(1, 100)
    quality = st.select_slider("Choose the quality of the final picture", options=my_range, value=70)
    if video_file is not None:
        capture = load_video(video_file)
        img = heatmap(capture, quality)
        st.image(img)




if __name__ == '__main__':
    main()