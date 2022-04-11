# Visualizing Sound In The Wild

[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@topiwala.anirudh/visualizing-sound-in-the-wild-b500657b0d85)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---
# Overview

### What if you could capture sound in an image? What if you could add another dimension to a still photograph? What if you could make a picture come alive?

This is what this projects aims to do. This is a project where the pictures are made more tangible by encoding the sound of the surroundings where the picture was taken into the image. Read more about how this works in my blog post **[*Visualizing Sound In The Wild*](https://medium.com/@topiwala.anirudh/visualizing-sound-in-the-wild-b500657b0d85)**.

To understand how the algorithm works and make the experience more tangible I create an app using Streamlit. The app is hosted by Streamlit Share and Heroku.
1. **[*Visualizing Sound In the Wild on Streamlit Share*](https://share.streamlit.io/anirudhtopiwala/visualize_sound_webapp/main/web_app.py)**  [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/anirudhtopiwala/visualize_sound_webapp/main/web_app.py)
2. **[*Visualizing Sound In the Wild on Heroku*](https://visualize-sound.herokuapp.com/)**  [![Heroku](https://img.shields.io/badge/heroku-%23430098.svg?style=for-the-badge&logo=heroku&logoColor=white)](https://visualize-sound.herokuapp.com/)


Users can visualize sound on any image of their choosing. The app supports two methods of visualizing sound:
1. Visualizing Sound in Real Time - Use your own voice to visualize sound.
2. Visualizing Sound Youtube - Use audio from any YouTube video to visualize sound.

This app also doubles as a way to extract and download audio from any YouTube video.

## Projects
Below are some projects that demonstrates how sound can be visualized in an image. The idea of visualizing sound is not limited to capturing the source of the sound but can be extended to capture any meaningful sound and visualize it in the picture. Think of it like tagging sound in an image instead of a location, but in a more visual way.

1. [Seattle Great Wheel](https://github.com/anirudhtopiwala/Visualize_Sound_In_The_Wild#Seattle-Great-Wheel)
2. [Instagram or Snapchat Filter?](https://github.com/anirudhtopiwala/Visualize_Sound_In_The_Wild#Instagram-or-Snapchat-Filter)
3. [Hiking Among The Trees](https://github.com/anirudhtopiwala/Visualize_Sound_In_The_Wild#Hiking-Among-The-Trees)

A complete list of projects can be found [here](https://github.com/anirudhtopiwala/Visualize_Sound_In_The_Wild/blob/main/projects/PROJECTS.md#Visualizing-Sound-In-The-Wild) and in the YouTube playlist [Visualize Sound Projects](https://youtube.com/playlist?list=PLwL956DTEkdkA0pt-GFHNQqifHVksmZjY).

### Seattle Great Wheel
Giving a 360 view of Seattle, the ferry wheel gives outstanding views of the city at 60 ft elevation right above the Puget Sound. Although, its real beauty shines up in the night. Right next to the wheel is the Historic Carousel and Wings Over Washington. These landmarks are the first thing that comes to mind and when thinking of Seattle and so I wanted to merge these into one picture. The sound from the carousel varies the brightness of the amazing lights on the wheel making the image come alive. Also don't forget to admire at the reflection of the lights in the water.

[![Image](https://img.youtube.com/vi/kIUXiqmtZqg/0.jpg)](https://www.youtube.com/watch?v=kIUXiqmtZqg)

### Instagram or Snapchat Filter?
This is a video of me with an existing Devil Horns Snapchat filter, but with my own way of encoding the music into the picture. See how the music [Some Nights](https://www.youtube.com/watch?v=qQkBeOisNM0) is encoded into the picture and makes the picture come alive. Does this look cool or what?  Would you want this as an Instagram or a Snapchat filter?

[![Image](https://img.youtube.com/vi/nDn1x1KHtOQ/0.jpg)](https://www.youtube.com/watch?v=nDn1x1KHtOQ)

### Hiking Among The Trees
An early morning hike on the Mirror Lake trail in the woods of Mount Hood, Oregon. Somewhere among the trees, I was taking a break and lying down in the snow. It was here, I captured this picture and recorded the sound of chirping birds. The sound from the birds vary the brightness of the sky making the image come alive.

[![Image](https://img.youtube.com/vi/XUllsgl0diw/0.jpg)](https://www.youtube.com/watch?v=XUllsgl0diw)

## Run the App Locally
If you want to make any changes to the code or were directed here cause the hosted apps on [Streamlit Share](https://share.streamlit.io/anirudhtopiwala/visualize_sound_webapp/main/web_app.py) and [Heroku](https://visualize-sound.herokuapp.com) are not working, here is a step by step guide on how to run the app locally.

Clone the repository locally.
```
git clone https://github.com/anirudhtopiwala/Visualize_Sound_In_The_Wild.git
```
Activate your virtual environment or create a new one. You can create a virtual environment by running:
```
python3 -m venv ${ENV_NAME}
source ${ENV_NAME}/bin/activate
```

Once your environment is active, install the dependencies and required packages by going to the repository directory and running:
```
pip install -r requirements.txt
```
Start the app by running:
```
streamlit run web_app.py
```

The advantage of running the app locally is that you get to visualize sound in real time and can also change the code to process youtube videos for a longer duration of time. This is not possible on the hosted apps because of limited resources.


## Contact Me
[Anirudh Topiwala](https://anirudhtopiwala.com/) [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/anirudhtopiwala/) [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anirudhtopiwala/) [![Twitter](https://img.shields.io/badge/<handle>-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/TopiwalaAnirudh)
