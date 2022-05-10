import time

import cv2
import numpy as np
from PIL import Image

from frcnn import FRCNN

if __name__ == "__main__":

    frcnn = FRCNN()

    # mode:
    # predict_img: predict a single image
    # predict_video: predict a video
    # predict_imgs: predict images in the folder
    mode = "predict_img"
    crop = False

    # Change them when predicting videos
    video_path = 0
    video_save_path = ""
    video_fps = 0

    fps_test_interval = 100

    img_path = "img/"
    img_save_path = "img_out/"

    if mode == "predict":

        while True:
            img = input("Enter the image name:")

            try:
                image = Image.open(img)
            except:
                print("Cannot open the file, try again!.")
                continue

            r_image = frcnn.detect_image(image, crop = crop)
            r_image.show()

    elif mode == "video":
        capture = cv2.Videocapture(video_path)
        if video_save_path != "":
            # choose the output video format
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_size = (w, h)
            out_video = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        prev_frame_time = 0
        new_frame_time = 0
        while(True):
            t1 = time.time()
            # ret here is a boolean that returns true if the frame is avalible
            ret, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(frcnn.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            new_frame_time = time.time()
            fps = int(1/ (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX,\
                                                1, (0, 255, 0), 2)
            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff

            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = frcnn.get_FPS(img, test_interval)
        print(str(tact_time) + 'seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1' )

    elif mode == "predict_imgs":
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', \
                                '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(img_path, img_name)
                image = Image.open(image_path)
                r_image = frcnn.detect_image(image)

                if not os.path.exists(img_save_path):
                    os.makedirs(img_save_path)

                r_image.save(os.path.join(img_save_path, img_name.replace(".jpg", ".png")), quality = 95, subsampling = 0)
    else:
        raise AssertionError("Please enter the correct mode!")
