---
title: Celebrity Lookalike Finder
emoji: 🎭
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# celebrity lookalike finder

this is a cool project that tells you which famous person you look like. it uses a webcam or you can just upload a pic and it finds your twin lol.

### how to run it
1. install the stuff: `pip install -r requirements.txt`
2. run the app: `python app.py`
3. go to `localhost:5000` in your browser

### how it works
it uses mediapipe face recognition to compare ur face to like a bunch of celebrities. that are stored in the celebs folder, if you want to add more, there is a feature in which you an name a celebrity and the web scraping program will find a photo of that celebrity and download it
