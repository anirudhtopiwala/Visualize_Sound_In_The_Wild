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
Visualizing Sound In The Wild is a project where the pictures are made more
tangible by encoding the sound of the surroundings where the picture was taken
into the image. This is a web app that lets users interact with the algorithm and
lets them visualize their own voice on nay image of their choosing.
Users can also encode audio from any YouTube video into the image.

Read more about this on my blog:

The app is hosted on:
1: Streamlit Share: https://share.streamlit.io/anirudhtopiwala/visualize_sound_webapp/main/web_app.py
2. Heroku: https://visualize-sound.herokuapp.com/
"""
import io
import logging
import multiprocessing
import os
import random
import tempfile

import altair as alt
import cv2
import numpy as np
import pandas as pd
import pydub
import streamlit as st
from moviepy.editor import AudioFileClip, ImageClip, VideoClip, concatenate_videoclips
from PIL import Image
from pytube import YouTube
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

logger = logging.getLogger(__name__)

sample_images = [
    "files/trees.jpg",
    "files/waterfall_orig.jpg",
    "files/seattle_wheel.jpg",
]
sample_images_mask = [
    "files/trees_mask.png",
    "files/waterfall_mask.png",
    "files/seattle_wheel_mask.png",
]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def adjust_brightness(img: np.array, adjust_brightness_value: float) -> np.array:
    """Adjusts the brightness of an image by converting it to HSV colorpace.

    img: Input RGB image.
    adjust_brightness_value: Value in range [0,1] by which the brightness will be adjusted.

    returns: A brightness adjusted RGB image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = img.astype(np.float32)
    img[:, :, 2] *= adjust_brightness_value
    img = np.uint8(img)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


@st.cache
def process_image(
    img: np.array, img_mask: np.array
) -> Tuple[np.array, np.array, np.array]:
    """Generates the image foreground and background using the input image and optionally provided image mask.

    It also rescales the images to a smaller size for faster processing. Additionally adds a sign to the image background.

    img: Input RGB image.
    img_mask: Optional image mask to create distinct image foreground and background.

    returns: A tuple of image foreground and image background generated using the image mask.
    """
    assert img is not None
    # Resize image if its too large.
    img_height, img_width, channels = img.shape
    max_size = 700
    while img_width > max_size or img_height > max_size:
        img_width = img_width / 2
        img_height = img_height / 2
    img_width = int(img_width)
    img_height = int(img_height)

    img = cv2.resize(img, (img_width, img_height))
    if img_mask is None:
        # If no mask is given, the given image becomes image foreground, as thats where the sound is encoded.
        img_foreground = img
        img_background = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    else:
        img_mask = cv2.resize(img_mask, (img_width, img_height))
        # Generate a binary image from the RGB mask.
        binary_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2GRAY) > 0
        binary_mask = np.uint8(binary_mask * 255)
        # Use the mask to generate a foreground and backgrund image. Sound will be encoded in the image foreground.
        img_foreground = cv2.bitwise_or(img, img, mask=binary_mask)
        img_background = cv2.bitwise_or(img, img, mask=cv2.bitwise_not(binary_mask))

    # Add Sign.
    sign_img = cv2.imread("files/sign.png", cv2.IMREAD_GRAYSCALE)
    sign_img = ((sign_img > 0) * 255).astype(np.uint8)
    # A scale of 0.15 usually works when working with images less than 500px range.
    scale = 0.15
    adjusted_height = int(sign_img.shape[0] * scale)
    adjusted_width = int(sign_img.shape[1] * scale)
    resized_sign_img = cv2.resize(sign_img, (adjusted_width, adjusted_height))
    binary_sign_img = np.zeros((img_height, img_width), dtype=np.uint8)
    binary_sign_img[
        img_height - adjusted_height - 10 : img_height - 10,
        img_width - adjusted_width - 10 : img_width - 10,
    ] = resized_sign_img
    #  Get rgb image.
    binary_sign_img = np.stack(
        (binary_sign_img, binary_sign_img, binary_sign_img), axis=2
    )
    # Watermark the sign.
    cv2.addWeighted(img_background, 1.0, binary_sign_img, 0.5, 0, img_background)
    return img, img_foreground, img_background


def draw_sound(img: np.array, amplitudes: np.array) -> None:
    """Draws the amplitude wave on the input image.

    img: Input RGB image.
    amplitudes: Array of amplitude values to plot.
    """
    img_height, img_width, _ = img.shape
    pos_x = 20
    pos_y = img_height - 50
    box_width_px = img_width / 4
    box_height_px = img_height / 5
    num_points = len(amplitudes)
    step_size_in_x = box_width_px / num_points
    step_size_in_y = box_height_px / 6
    line_thickness = 2
    color = (255, 255, 255)
    for i in range(0, len(amplitudes) - 2):
        point1 = (
            round(pos_x + (i * step_size_in_x)),
            round(pos_y - (amplitudes[i].item() * step_size_in_y)),
        )
        point2 = (
            round(pos_x + ((i + 1) * step_size_in_x)),
            round(pos_y - (amplitudes[i + 1].item() * step_size_in_y)),
        )
        cv2.line(img, point1, point2, color, thickness=line_thickness)


def encode_image(
    amplitudes_per_img_frame: np.array,
    img_foreground: np.array,
    img_background: np.array,
    should_plot: bool = False,
) -> np.array:
    """Takes the absolute maximum value from the given amplitudes and adjusts the image brightness to encode sound in the image.

    amplitudes_per_img_frame: Amplitudes to encode per image frame.
    img_foreground: Image foreground whose brightness will be adjusted.
    img_background: Image background.
    should_plot: Optionally plot the sound wave on the image.

    returns: Image formed by image foreground and img background with encoded amplitudes.
    """
    # Max amplitude represents maximum deviation of brightness.
    max_val = max(amplitudes_per_img_frame, key=abs)
    # This is not usually required, although clipping to remove noise.
    # If trying to recreate the original sound wave from the encoded image, comment the below line.
    max_val = np.clip(max_val, -0.3, 0.3)
    # Negative wave usually has a stronger amplitude from experimenting and increasing brightness is
    # visually more appealing than reducing it.
    max_val = -max_val
    img_foreground = adjust_brightness(img_foreground, max_val + 0.7)
    merged_image = np.add(img_foreground, img_background)
    if should_plot:
        draw_sound(merged_image, amplitudes_per_img_frame)
    return np.asarray(merged_image, dtype=np.uint8)


def load_image() -> Tuple[np.array, np.array]:
    """Streamlit configuration to load the image and optionally the image mask.

    returns: The loaded image and optional image mask.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("""Please upload an image or use an existing example.""")
    img = None
    img_mask = None

    # Upload an Image or use the defaults.
    uploaded_img_file = st.sidebar.file_uploader(
        "Choose a File",
        type=["png", "jpg", "jpeg"],
    )
    # Optionally upload image mask to restrict area in which sound is visualized.
    uploaded_img_mask = st.sidebar.file_uploader(
        "Optionally upload an image mask to restrict the area in which sound is visualized. (Size should match that of input image)",
        type=["png", "jpg", "jpeg"],
    )

    if st.sidebar.button("Use an existing example"):
        # Choose a random sample image.
        st.sidebar.write("Click again to try out a different image.")
        random_index = random.randint(0, len(sample_images) - 1)
        img = Image.open(sample_images[random_index])
        img_mask = Image.open(sample_images_mask[random_index])
    else:
        if uploaded_img_file is not None:
            img = Image.open(io.BytesIO(uploaded_img_file.getvalue()))
            img = img.convert("RGB")

        if uploaded_img_mask is not None:
            img_mask = Image.open(io.BytesIO(uploaded_img_mask.getvalue()))
            img_mask = img_mask.convert("RGB")

        if img and img_mask and img.size != img_mask.size:
            st.warning(
                f"Image mask of size {img_mask.size} does not match input img size {img.size}. Please upload mask that is the same size as image."
            )
            st.stop()

    if img is None and img_mask:
        st.warning(
            "Image mask only works with an uploaded image. Please upload an image using the sidebar or use an existing example."
        )
        st.stop()
    elif img is None:
        st.warning(
            "Please use the sidebar to upload an image or use an existing example. The sidebar can be expanded from the top left corner."
        )
        st.stop()

    # Conver PIL image to array.
    img = np.asarray(img, dtype=np.uint8)

    # Check that img and img_mask are rgb images.
    if img is not None and img.shape[2] != 3:
        st.warning(
            f"Image has to be an RGB image. Uploaded image has {img.shape[2]} channels. Please upload another image or use an existing example."
        )
        st.stop()

    if img_mask:
        img_mask = np.asarray(img_mask, dtype=np.uint8)
        if img_mask.shape[2] != 3:
            st.warning(
                f"Image mask has to be an RGB image. Uploaded image has {img.shape[2]} channels. If you have a binary image please concatenate the channels to make it RGB."
            )
            st.stop()

    # Show the images in sidebar.
    st.sidebar.write("Image loaded successfully.")

    if img is not None:
        st.sidebar.write("Main Image")
        st.sidebar.image(img)
    if img_mask is not None:
        st.sidebar.write("Image Mask")
        st.sidebar.image(img_mask)

    return img, img_mask


# A TTL of 120 seconds helps with running the app faster and avoid OOM error.
@st.cache(suppress_st_warning=True, ttl=120)
def load_audio_from_link(yt_link: str) -> pydub.AudioSegment:
    """Uses pytube to extract audio from YouTube video.

    Value error is raised if the youtube link is invalid.

    yt_link: Youtube link to load audio from.

    returns: Single channel audio extracted from the video.
    """
    try:
        yt = YouTube(yt_link)
    except VideoUnavailable as e:
        return e
    strm = yt.streams.filter(only_audio=True, file_extension="mp4").first()
    if strm is None:
        return ValueError("Unable to load link.")

    # Reset the buffer and get the audio.
    buff = io.BytesIO()
    strm.stream_to_buffer(buff)
    buff.seek(0)
    full_audio = pydub.AudioSegment.from_file(buff)
    mono_audio = full_audio.split_to_mono()[0]
    return mono_audio


def get_youtube_link() -> str:
    """Streamlit selectbox to get youtube link from user."""
    yt_audios = {
        "Imagine Dragons: Believer": "https://www.youtube.com/watch?v=Roi4TG6ZvKk",
        "You are my Sunshine": "https://www.youtube.com/watch?v=dh7LJDHFaqA",
        "Doobey": "https://www.youtube.com/watch?v=6eGCi4SVy94",
        "Lindsey Stirling - Crystallize": "https://www.youtube.com/watch?v=aHjpOzsQ9YI",
    }
    select_box_link = st.selectbox(
        "Choose a song or use your own link.", yt_audios.keys()
    )
    link = st.text_input("YouTube Link", yt_audios[select_box_link])
    st.write(f"Using YouTube link: {link}.")
    return link


def visualize_youtube_video() -> None:
    """Main function function which sets up the "Visualize Sound - Youtube" page of the app.

    It has the following function:
    1. Takes user given youtube link or user selection of an existing link.
    2. Takes image and optionally image mask from the user.
    3. Extracts audio from the loaded youtube video.
    4. Allows user to specify a fps and time duration of the video to extract audio from.
    5. Encodes the audio into the image.
    6. Allowed the user to download the extracted audio.
    """
    st.header("Visualizing Sound Using Audio from YouTube")
    st.write(
        """The app takes a YouTube video as a link, extracts the audio for the given time interval and renders the video by visualizing sound in the image of your choice."""
    )
    st.write(
        """Feel free to select an option from the dropdown menu for a quick demo or paste a youtube link in the textbox below."""
    )
    link = get_youtube_link()
    try:
        audio = load_audio_from_link(link)
    except:
        st.warning(
            "The video is unavailable, please try reloading or using a different link. If the issue persists, try running the [app locally](https://github.com/anirudhtopiwala/visualize_sound_webapp)"
        )
        st.stop()

    # Get the time span of the audio and set the range selection sliders.
    max_time_s = 10
    durations_seconds = int(audio.duration_seconds)
    start_time, end_time = st.select_slider(
        f"Woahh found {durations_seconds} seconds of audio!!! Please select a time interval within {max_time_s} s.",
        options=range(durations_seconds),
        value=(0, max_time_s),
    )
    audio_time = end_time - start_time
    if audio_time < 0 or audio_time > max_time_s:
        st.warning(f"Please reduce the selected range to less than {max_time_s}s.")
        st.stop()

    # Set the frame rate for the video.
    fps = st.radio(
        "Available frame rates (frames/second) for rendering the video.",
        (30, 60, 120),
        index=1,
    )
    st.write(
        "A higher frame rate allows visualizing more amplitudes and therefore would result in more fluctuations in the image, which is fun to see."
    )
    st.write(
        "Read more on how this works on my blog post ** *Visualizing Sound In The Wild* ** [![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/p/b500657b0d85/edit)"
    )

    # Get image arrays from user.
    img, img_mask = load_image()
    resized_img, img_foreground, img_background = process_image(img, img_mask)

    # Cut the aduio to the specified range.
    cut_audio = audio[start_time * 1000 : end_time * 1000]
    st.write("*Give the audio a listen while we get the visualization ready...*")
    st.write(cut_audio)

    # Temp file for writing the final video.
    with tempfile.NamedTemporaryFile("w+b", suffix=".mp4") as video_writer:
        with st.spinner("Encoding Sound in the image..."):
            # Update images using FPS:
            chunk_ms = (1 / fps) * 1000
            chunks = pydub.utils.make_chunks(cut_audio, chunk_ms)

            # Get Image Clips
            img_clips = []
            for chunk in chunks:
                # Dividing the amplitude by 10000 to get values in range [-1, 1].
                # Original amplitudes can still be retrieved back as scaling the
                # amplitudes still preserves the original signal.
                sound_array = np.array(chunk.get_array_of_samples()) / 10000
                img = encode_image(sound_array, img_foreground, img_background, True)
                img_clips.append(ImageClip(img).set_duration(1 / fps))

            # Create video reader from moviepy
            # Export the current audio clip to binary file.
            with tempfile.NamedTemporaryFile("w+b", suffix=".wav") as audio_writer:
                cut_audio.export(audio_writer.name, "wav")
                audio_clip = AudioFileClip(audio_writer.name)
                video_clip = concatenate_videoclips(img_clips, method="compose")
                video_clip_with_audio = video_clip.set_audio(audio_clip)
                video_clip_with_audio.write_videofile(
                    video_writer.name, fps=fps, threads=multiprocessing.cpu_count()
                )

        # Show the images
        col1, col2 = st.columns(2)
        with col1:
            st.image(resized_img)
        with col2:
            st.video(video_writer.name)

    st.write(
        "Try out different youtube videos to see how a shrill/coarse or loud/quite note affects the generated visualization."
    )
    st.subheader("Extract Audio from YouTube Video")
    st.write(
        "This app also doubles as a way to extract audio from any YouTube video. So here is the complete audio for you to download..."
    )
    st.warning("Audio may be subject to copyright.")
    st.write(audio)

    # Contact Me
    st.markdown("### Contact Me")
    st.markdown(
        "[Anirudh Topiwala](https://anirudhtopiwala.com/) [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/anirudhtopiwala) [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anirudhtopiwala/) [![Twitter](https://img.shields.io/badge/<handle>-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/TopiwalaAnirudh)"
    )


def visualize_sound_in_realtime() -> None:
    """Main function function which sets up the "Visualize Sound - RealTime" page of the app.

    It has the following function:
    1. Takes image and optionally image mask from the user.
    2. Extracts realtime audio from the users browser microphone. This is achieved by using webrtc.
    3. Encodes the audio into the image.
    6. Plots Amplitude vs Time for extracted audio from the user.
    """
    st.header("Visualizing Sound In Real Time")
    st.markdown("""Visualize your own voice on a picture of your liking.""")

    # Plot sound and see its effects on an image.
    samplerate = 48000  # Default samplerate for most microphones.
    # Visualization is only possible on mono audio.
    num_channels = 1
    #  Set up the webrtc stramer to get audio.
    webrtc_ctx = webrtc_streamer(
        key="visualize-sound",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=samplerate,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )

    if not webrtc_ctx.state.playing:
        return

    # Get image arrays from user.
    img, img_mask = load_image()
    resized_img, img_foreground, img_background = process_image(img, img_mask)

    # Setting up the page.
    status_indicator = st.empty()
    status_indicator.write("Running. Say something!")
    st.warning(
        "The app is known to freeze up when many people are using the app, if the images are not being updated in realtime consider reloading the app. Or select the option to visualize audio from a YouTube video in the sidebar. Another option is to run the [app locally](https://github.com/anirudhtopiwala/visualize_sound_webapp)."
    )
    st.write(
        "Notice how when you speak loudly the higher amplitude sounds waves generates a very bright or dark image."
    )
    st.write(
        "Now try whistling, the higher frequency of the whistling sound causes an increase in the rate of fluctuations of the brightness."
    )
    st.write(
        "Read more on how this works on my blog post ** *Visualizing Sound In the Wild* ** [![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/p/b500657b0d85/edit)"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_img)
    with col2:
        encoded_image_st = st.empty()
    st.markdown(
        "<h5 style='text-align: center;'>Amplitude vs Time</h5>", unsafe_allow_html=True
    )
    fig_st = st.empty()
    st.write(
        "Try out different examples by clicking on ** *'Use an existing example'* ** in the sidebar. Or upload your own image."
    )

    # Contact Me
    st.markdown("### Contact Me")
    st.markdown(
        "[Anirudh Topiwala](https://anirudhtopiwala.com/) [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/anirudhtopiwala) [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anirudhtopiwala/) [![Twitter](https://img.shields.io/badge/<handle>-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/TopiwalaAnirudh)"
    )

    while True:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except:
            status_indicator.write(
                "No frame arrived. Please check audio permissions and reload the page."
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
        # Dividing the amplitude by 10000 to get values in range [-1, 1].
        # Original amplitudes can still be retrieved back as scaling the
        # amplitudes still preserves the original signal.
        sound_array = np.array(sound.get_array_of_samples()) / 10000

        encoded_image_st.image(
            encode_image(sound_array, img_foreground, img_background)
        )

        # Visualize the last extracted sound frame.
        times = (np.arange(-len(sound_array), 0)) / sound.frame_rate
        amplitude_df = pd.DataFrame({"Amplitude": sound_array, "Time": times})

        fig_st.altair_chart(
            alt.Chart(amplitude_df)
            .mark_line()
            .encode(
                alt.Y("Amplitude", scale=alt.Scale(domain=(-1.5, 1.5))),
                alt.X("Time", axis=None),
            ),
            use_container_width=True,
        )


def welcome() -> None:
    """Sets up the Welcome page of the app."""
    st.title("Visualizing Sound In The Wild")
    st.subheader("A simple app that lets you visualize sound in an image.")

    st.write(
        "What if you could capture sound in an image? What if you could add another dimension to a still photograph?"
        " What if you can make taking pictures more tangible?"
    )
    st.markdown(
        "This is what this projects aims to do. Read more on how this is done on my blog post:"
    )
    st.markdown(
        "** *Visualizing Sound In The Wild* ** [![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/p/b500657b0d85/edit)"
    )
    st.markdown(
        """
        The app supports two methods of visualizing sound:
        * Visualizing Sound in Real Time - Use your own voice to visualize sound.
        * Visualizing Sound Youtube - Use audio from any YouTube video to visualize sound.
        """
    )
    st.write(
        "These options can be selected in the sidebar which is available in the top left corner of the screen."
    )

    # Play examples !!!
    st.subheader("Examples in the Wild")
    # Hiking Among The Trees
    st.markdown("#### Hiking Among The Trees")
    st.markdown(
        "*An early morning hike on the Mirror Lake trail in the woods of Mount Hood, Oregon*"
    )
    st.video("https://youtu.be/XUllsgl0diw")

    # The Mighty Multnomah Falls
    st.markdown("#### The Mighty Multnomah Falls")
    st.markdown(
        "*Being the tallest waterfall in Oregon, listen in on 923 gallons of water falling over a height of 620 feet every second.*"
    )
    st.video("https://youtu.be/AqIaOgdsWUo")

    # Seattle Great Wheel
    st.markdown("#### Seattle Great Wheel")
    st.markdown(
        "*A 360 view of Seattle, the ferry wheel gives outstanding views of the city at 60ft directly above the Puget Sound. Listen in on how the sounds from a carousel next to the wheel lights up the colors in the night.*"
    )
    st.video("https://youtu.be/kIUXiqmtZqg")

    # Contact Me
    st.markdown("### Contact Me")
    st.markdown(
        "[Anirudh Topiwala](https://anirudhtopiwala.com/) [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/anirudhtopiwala) [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anirudhtopiwala/) [![Twitter](https://img.shields.io/badge/<handle>-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/TopiwalaAnirudh)"
    )


def main() -> None:
    """Sets up the options to navigate between different pages."""
    selected_box = st.sidebar.selectbox(
        "Choose one of the following",
        (
            "Project Overview",
            "Visualize Sound in Real Time",
            "Visualize Sound - YouTube",
        ),
    )

    if selected_box == "Project Overview":
        welcome()
    elif selected_box == "Visualize Sound in Real Time":
        visualize_sound_in_realtime()
    elif selected_box == "Visualize Sound - YouTube":
        visualize_youtube_video()


if __name__ == "__main__":
    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )
    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)
    main()
