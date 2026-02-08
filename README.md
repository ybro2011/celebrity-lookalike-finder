# celebrity lookalike

finds which celebrity you look like

## how to run

```bash
pip install -r requirements.txt
python app.py
```

then go to http://localhost:5000

## hosting

use render.com or railway.app (github pages doesnt work for flask apps)

on render:
- new web service
- connect github repo
- build: `chmod +x build.sh && ./build.sh`
- start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- note: free tier might run out of memory, may need to upgrade to $7/month starter plan

## stuff

- put celeb photos in the `celebs` folder
- uses face recognition to match
- webcam or upload photo

