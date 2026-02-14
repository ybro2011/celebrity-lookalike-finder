import os
import requests
import cv2
import mediapipe as mp
import numpy as np
import time
import random
import sys
import json
import pickle
from ddgs import DDGS

TMDB_API_KEY = os.environ.get("TMDB_API_KEY") 

class CelebScraper:
    def __init__(self):
        self.base_dir = "celebs"
        if not os.path.exists(self.base_dir): os.makedirs(self.base_dir)
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.6
        )
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def is_face_present(self, image_bytes):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: return False
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_img)
            return results.detections is not None
        except: return False

    def get_tmdb_image(self, name):
        try:
            url = f"https://api.themoviedb.org/3/search/person"
            params = {"api_key": TMDB_API_KEY, "query": name}
            data = requests.get(url, params=params, timeout=5).json()
            if data.get('results'):
                path = data['results'][0].get('profile_path')
                if path:
                    img_url = f"https://image.tmdb.org/t/p/h632{path}"
                    resp = requests.get(img_url, timeout=10)
                    if resp.status_code == 200: return resp.content
            return None
        except: return None

    def get_wikipedia_image(self, name):
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query", "titles": name, "prop": "pageimages",
                "format": "json", "pithumbsize": 1000, "redirects": 1
            }
            resp = requests.get(url, params=params, timeout=5).json()
            pages = resp.get("query", {}).get("pages", {})
            for p in pages.values():
                if "thumbnail" in p:
                    img_resp = requests.get(p["thumbnail"]["source"], timeout=10)
                    if img_resp.status_code == 200: return img_resp.content
            return None
        except: return None

    def draw_progress_bar(self, current, total, status_text=""):
        fraction = current / total
        bar = ('=' * int(fraction * 30)).ljust(30)
        sys.stdout.write(f'\rProgress: [{bar}] {current}/{total} ({int(fraction*100)}%) | {status_text[:30].ljust(30)}')
        sys.stdout.flush()

    def scrape(self, celeb_list):
        added = 0
        total = len(celeb_list)
        
        for idx, name in enumerate(celeb_list):
            clean_name = name.replace(" ", "_").lower()
            existing = [f for f in os.listdir(self.base_dir) if f.lower().startswith(clean_name)]
            if existing:
                self.draw_progress_bar(idx + 1, total, f"skipped {name}")
                continue
            
            img_data = None
            img_data = self.get_tmdb_image(name)
            if img_data and not self.is_face_present(img_data):
                img_data = None
            
            if not img_data:
                img_data = self.get_wikipedia_image(name)
                if img_data and not self.is_face_present(img_data):
                    img_data = None
            
            if not img_data:
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.images(keywords=f"{name} headshot portrait", max_results=2))
                        for r in results:
                            resp = requests.get(r['image'], timeout=5, headers=self.headers)
                            if resp.status_code == 200 and self.is_face_present(resp.content):
                                img_data = resp.content
                                break
                except:
                    pass
            
            if img_data and self.is_face_present(img_data):
                filename = f"{clean_name}_{int(time.time())}.jpg"
                filepath = os.path.join(self.base_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                added += 1
                self.draw_progress_bar(idx + 1, total, f"added {name}")
            else:
                self.draw_progress_bar(idx + 1, total, f"failed {name}")
            
            time.sleep(0.5)
        
        print("\nscrape complete!")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config", "celebs.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    CelebScraper().scrape(config["celebs"])