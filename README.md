# Visualizing Sound In The Wild

[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@topiwala.anirudh/visualizing-sound-in-the-wild-b500657b0d85)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---
# Overview

### What if you could capture sound in an image? What if you could add another dimension to a still photograph? What if you could make a picture come alive?

This is what this projects aims to do. This is a project where the pictures are made more tangible by encoding the sound of the surroundings where the picture was taken into the image. Read more about how this works in my blog post **[*Visualizing Sound In The Wild*](https://medium.com/@topiwala.anirudh/visualizing-sound-in-the-wild-b500657b0d85)**.

To understand how the algorithm works and make the experience more tangible I create an app using Streamlit. The app is hosted by Streamlit Share and Heroku.
1. **[*Visualizing Sound In the Wild on Streamlit Share*](https://share.streamlit.io/anirudhtopiwala/visualize_sound_in_the_wild/main/web_app.py)**  [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/anirudhtopiwala/visualize_sound_in_the_wild/main/web_app.py)
2. **[*Visualizing Sound In the Wild on Heroku*](https://visualize-sound.herokuapp.com/)**  [![Heroku](https://img.shields.io/badge/heroku-%23430098.svg?style=for-the-badge&logo=heroku&logoColor=white)](https://visualize-sound.herokuapp.com/)


Users can visualize sound on any image of their choosing. The app supports two methods of visualizing sound:
1. Visualizing Sound in Real Time - Use your own voice to visualize sound.
2. Visualizing Sound Youtube - Use audio from any YouTube video to visualize sound.

This app also doubles as a way to extract and download audio from any YouTube video.

## Projects
Below are some projects that demonstrates how sound can be visualized in an image. The idea of visualizing sound is not limited to capturing the source of the sound but can be extended to capture any meaningful sound and visualize it in the picture. Think of it like tagging sound in an image instead of a location, but in a more visual way.

1. [The Mighty Multnomah Falls](https://github.com/anirudhtopiwala/Visualize_Sound_In_The_Wild#The-Mighty-Multnomah-Falls)
2. [Hiking Among The Trees](https://github.com/anirudhtopiwala/Visualize_Sound_In_The_Wild#Hiking-Among-The-Trees)
3. [Instagram or Snapchat Filter?](https://github.com/anirudhtopiwala/Visualize_Sound_In_The_Wild#Instagram-or-Snapchat-Filter)


***A complete list of projects can be found [here](https://github.com/anirudhtopiwala/Visualize_Sound_In_The_Wild/blob/main/projects/PROJECTS.md#Visualizing-Sound-In-The-Wild) and in the YouTube playlist [Visualize Sound Projects](https://youtube.com/playlist?list=PLwL956DTEkdkA0pt-GFHNQqifHVksmZjY)***.

## The Mighty Multnomah Falls
Being the tallest waterfall in Oregon, listen in on 923 gallons of water falling over a height of 620 feet every second. The sound from the waterfall varies the brightness of the waterfall making the image come alive.

[![Image](https://img.youtube.com/vi/YdY4I7n0Cpw/0.jpg)](https://www.youtube.com/watch?v=YdY4I7n0Cpw)

Get exclusive access to the original HD picture along with the sound of Multnomah Falls used to make this art. <a href="https://opensea.io/assets/0x495f947276749ce646f68ac8c248420045cb7b5e/46670013259754365806976296485688128176995984954052073858673724124976090447873/" title="Buy on OpenSea" target="_blank"><img style="width:55px; border-radius:2px; box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.25);" src="https://storage.googleapis.com/opensea-static/Logomark/Badge%20-%20Available%20On%20-%20Dark.png" alt="OpenSea" /></a>


### Hiking Among The Trees
An early morning hike on the Mirror Lake trail in the woods of Mount Hood, Oregon. Somewhere among the trees, I was taking a break and lying down in the snow. It was here, I captured this picture and recorded the sound of chirping birds. The sound from the birds vary the brightness of the sky making the image come alive.

[![Image](https://img.youtube.com/vi/XUllsgl0diw/0.jpg)](https://www.youtube.com/watch?v=XUllsgl0diw)

Get exclusive access to the original HD picture along with the sound of the chirping birds used to make this art. <a href="https://opensea.io/assets/0x495f947276749ce646f68ac8c248420045cb7b5e/46670013259754365806976296485688128176995984954052073858673724126075602075649/" title="Buy on OpenSea" target="_blank"><img style="width:55px; border-radius:2px; box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.25);" src="https://storage.googleapis.com/opensea-static/Logomark/Badge%20-%20Available%20On%20-%20Dark.png" alt="OpenSea" /></a>

### Instagram or Snapchat Filter?
This is a video of me with an existing Devil Horns Snapchat filter, but with my own way of encoding the music into the picture. See how the music [Some Nights](https://www.youtube.com/watch?v=qQkBeOisNM0) is encoded into the picture and makes the picture come alive. Does this look cool or what?  Would you want this as an Instagram or a Snapchat filter?

[![Image](https://img.youtube.com/vi/nDn1x1KHtOQ/0.jpg)](https://www.youtube.com/watch?v=nDn1x1KHtOQ)

## Run the App Locally
If you want to make any changes to the code or were directed here cause the hosted apps on [Streamlit Share](https://share.streamlit.io/anirudhtopiwala/visualize_sound_in_the_wild/main/web_app.py) and [Heroku](https://visualize-sound.herokuapp.com) are not working, here is a step by step guide on how to run the app locally.

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
[Anirudh Topiwala](https://anirudhtopiwala.com/)
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/anirudhtopiwala/)
[![YouTube](https://img.shields.io/badge/Visualizing_Sound_In_The_Wild-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/channel/UCFKaLmO8K11veL8JJZNPR1Q)
[![Instagram](https://img.shields.io/badge/visualize_sound-%23E4405F.svg?style=for-the-badge&logo=Instagram&logoColor=white)](https://www.instagram.com/visualize_sound/)
<a href="https://opensea.io/collection/visualize-sound" title="Buy on OpenSea" target="_blank"><img style="width:80px; border-radius:2px; box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.25);" src="https://storage.googleapis.com/opensea-static/Logomark/Badge%20-%20Available%20On%20-%20Dark.png" alt="OpenSea" /></a>
