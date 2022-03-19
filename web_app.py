# Copyright 2022 Anirudh Topiwala

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A web application to visualize sound in images.
"""
import io
import logging
import random

import altair as alt
import cv2
import numpy as np
import pandas as pd
import pydub
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
from pytube import YouTube

logger = logging.getLogger(__name__)

sample_images = ["files/trees.jpg", "files/waterfall_orig.jpg"]
sample_images_mask = ["files/trees_mask.png", "files/waterfall_mask.png"]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{
        "urls": ["stun:stun.l.google.com:19302"]
    }]})


@st.cache
def adjust_brightness(img, value):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_img = hsv_img.astype(np.float32)
    hsv_img /= 255
    hsv_img[:, :, 2] = hsv_img[:, :, 2] * value
    hsv_img = np.uint8(hsv_img * 255)
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)


@st.cache
def process_image(img, img_mask):
    height, width, _ = img.shape
    binary_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2GRAY) > 0
    binary_mask = np.uint8(binary_mask * 255)

    # Sound variations will be added to image foreground.
    img_foreground = cv2.bitwise_or(img, img, mask=binary_mask)
    img_background = cv2.bitwise_or(img,
                                    img,
                                    mask=cv2.bitwise_not(binary_mask))

    # Add Sign.
    sign_img = cv2.imread("files/sign.png", cv2.IMREAD_GRAYSCALE)
    sign_img = ((sign_img > 0) * 255).astype(np.uint8)
    scale = 0.25
    adjusted_height = int(sign_img.shape[0] * scale)
    adjusted_width = int(sign_img.shape[1] * scale)
    resized_sign_img = cv2.resize(sign_img, (adjusted_width, adjusted_height))
    empty_img = np.zeros((height, width), dtype=np.uint8)
    empty_img[height - adjusted_height - 10:height - 10,
              width - adjusted_width - 10:width - 10] = resized_sign_img
    #  Get rgb image.
    empty_img = np.stack((empty_img, empty_img, empty_img), axis=2)
    # Watermark the sign.
    cv2.addWeighted(img_background, 1.0, empty_img, 0.5, 0, img_background)

    return img_foreground, img_background


@st.cache
def encode_image(amplitudes_per_img_frame, img_foreground, img_background):
    max_val = max(amplitudes_per_img_frame, key=abs)
    max_val = np.clip(max_val, -0.3, 0.3)
    max_val = -max_val
    img_foreground_adjusted = adjust_brightness(img_foreground,
                                                min(max_val + 0.7, 1))
    merged_image = img_foreground_adjusted + img_background
    return np.asarray(merged_image, dtype=np.uint8)


def load_image():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """Please upload an image or use an exisiting example.""")
    img = None
    img_mask = None

    if st.sidebar.button("Use an existing example"):
        # Choose a random sample image.
        random_index = random.randint(0, len(sample_images) - 1)
        img = Image.open(sample_images[random_index])
        img_mask = Image.open(sample_images_mask[random_index])

    # Upload an Image or use the defaults.
    uploaded_img_file = st.sidebar.file_uploader(
        "Choose a File",
        type=["png", "jpg", "jpeg"],
    )
    # Optionally upload image mask to restrict area in which sound is visualized.
    uploaded_img_mask = st.sidebar.file_uploader(
        "Optionally upload a mask to restrict area in which sound is visualized. (Size should match that of input image)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_img_file is not None:
        img = Image.open(io.BytesIO(uploaded_img_file.getvalue()))

    if uploaded_img_mask is not None:
        img_mask = Image.open(io.BytesIO(uploaded_img_mask.getvalue()))

    if img and img_mask and img.size != img_mask.size:
        st.warning(
            f"Image mask of size {img_mask.size} does not macth input img size {img.size}. Please upload mask that is the same size as image."
        )
        st.stop()

    if not img:
        return None, None, None

    # Resize image if its too large.
    img_width, img_height = img.size
    while img_width > 1000 or img_height > 1000:
        img_width = int(img_width / 2)
        img_height = int(img_height / 2)
    img = img.resize((img_width, img_height))
    img_array = np.asarray(img, dtype=np.uint8)
    if img_mask:
        img_mask = img_mask.resize((img_width, img_height))
        img_mask = np.asarray(img_mask, dtype=np.uint8)
    else:
        # Empty mask.
        img_mask = np.zeros((img_width, img_height, 3), dtype=np.uint8)

    img_foreground, img_background = process_image(img_array, img_mask)
    return img, img_foreground, img_background


def get_sound(col1, col2, fps=60):
    samplerate = 48000
    # Num Channels.
    num_channels = 1
    webrtc_ctx = webrtc_streamer(
        key="visualize-sound",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=samplerate,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": False,
            "audio": True
        },
        async_processing=True,
    )

    if not webrtc_ctx.state.playing:
        return

    # Get image arrays from user.
    pil_img, img_foreground, img_background = load_image()
    if pil_img is None:
        st.warning("Please upload an image or use an exisiting example.")
        st.stop()

    status_indicator = st.empty()
    status_indicator.write("Running. Say something!")

    with col1:
        st.image(pil_img)
    with col2:
        encoded_image_st = st.empty()

    fig_st = st.empty()
    while True:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except:
            status_indicator.write(
                "No frame arrived. Please check audio permissions and refresh."
            )
            return

        sound = pydub.AudioSegment.empty()
        for audio_frame in audio_frames:
            sound += pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=num_channels,
            )
        # Dividing the ampltide by 10000 to get vaues in range [-1, 1]
        sound_array = np.array(sound.get_array_of_samples()) / 10000

        encoded_image_st.image(
            encode_image(sound_array, img_foreground, img_background))

        times = range(len(sound_array))
        source = pd.DataFrame({'Amplitude': sound_array, 'Time': times})

        fig_st.altair_chart(alt.Chart(source).mark_line().encode(
            alt.Y("Amplitude", scale=alt.Scale(domain=(-1.5, 1.5))),
            alt.X("Time", axis=None)),
                            use_container_width=True)


def main():
    selected_box = st.sidebar.selectbox('Choose one of the following', (
        'Welcome',
        'Visualize Sound in Real Time',
        'Visualize Sound - YouTube',
    ))

    if selected_box == 'Welcome':
        welcome()
    elif selected_box == 'Visualize Sound in Real Time':
        visualize_sound()
    elif selected_box == 'Visualize Sound - YouTube':
        visualize_youtube_video()


def visualize_youtube_video():
    st.header("Visualizing Sound !!!")
    st.markdown("""A first of its kind visualization of sound on an image.""")
    link = "https://www.youtube.com/watch?v=724BgFjKM08&list=RDMM&index=4"
    yt=YouTube(link)
    strm= yt.streams.filter(only_audio=True, file_extension='mp4').first()
    if strm is not None:
        buff = io.BytesIO()
        strm.stream_to_buffer(buff)
        buff.seek(0)
        full_audio = pydub.AudioSegment.from_file(buff)
        st.write(full_audio)






def visualize_sound():
    st.header("Visualizing Sound !!!")
    st.markdown("""A first of its kind visualization of sound on an image.""")

    # Plot sound and see its effects on an image.
    # Show the original image, image with effects and sound plot.
    col1, col2 = st.columns(2)
    get_sound(col1, col2)


def welcome():

    st.title("Visualizing Sound !!!")
    st.subheader('A simple app that lets you visualize sound in an image.')

    # Play an example !!!
    st.subheader('A day in the forest...')
    video_file = open('files/tree_with_sound.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG",
                           "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format=
        "[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
