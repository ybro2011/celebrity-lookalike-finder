#!/usr/bin/env python3
import os
import requests
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import random
from ddgs import DDGS
from PIL import Image
import io
import traceback

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

def is_face_present(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        )
        results = face_detector.process(rgb_img)
        has_face = results.detections is not None and len(results.detections) > 0
        return has_face
    except Exception as e:
        print(f"is_face_present error: {e}", flush=True)
        return False

def get_tmdb_image(name):
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
    except:
        return None

def get_wikipedia_image(name):
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
    except:
        return None

def seed_celebs():
    celebs_dir = "celebs"
    if not os.path.exists(celebs_dir):
        os.makedirs(celebs_dir, exist_ok=True)
    
    config_path = os.path.join(os.path.dirname(__file__), "config", "celebs.json")
    try:
        with open(config_path, "r") as f:
            celebs = json.load(f)["celebs"]
    except FileNotFoundError:
        print(f"config not found, using defaults")
        celebs = [
            "Taylor Swift", "Ariana Grande", "Justin Bieber", "Dua Lipa", "Rihanna",
            "Beyonce", "Lady Gaga", "Drake", "The Weeknd", "Zendaya",
            "Tom Holland", "Robert Downey Jr", "Scarlett Johansson", "Dwayne Johnson",
            "Tom Cruise", "Brad Pitt", "Leonardo DiCaprio", "Margot Robbie",
            "Jennifer Lawrence", "Ryan Reynolds"
        ]
    except Exception as e:
        print(f"error loading config: {e}, using defaults")
        celebs = [
            "Taylor Swift", "Ariana Grande", "Justin Bieber", "Dua Lipa", "Rihanna",
            "Beyonce", "Lady Gaga", "Drake", "The Weeknd", "Zendaya",
            "Tom Holland", "Robert Downey Jr", "Scarlett Johansson", "Dwayne Johnson",
            "Tom Cruise", "Brad Pitt", "Leonardo DiCaprio", "Margot Robbie",
            "Jennifer Lawrence", "Ryan Reynolds"
        ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    added = 0
    print("seeding celebrities...")
    
    for name in celebs:
        clean_name = name.replace(" ", "_").lower()
        
        existing = [f for f in os.listdir(celebs_dir) if f.lower().startswith(clean_name)]
        if existing:
            print(f"skipping {name} - already exists", flush=True)
            continue
        
        img_data = None
        
        print(f"trying {name}...", flush=True)
        img_data = get_tmdb_image(name)
        if img_data:
            print(f"  got TMDB image for {name}", flush=True)
            if not is_face_present(img_data):
                print(f"  no face in TMDB image for {name}", flush=True)
                img_data = None
        else:
            print(f"  no TMDB image for {name}", flush=True)
        
        if not img_data:
            img_data = get_wikipedia_image(name)
            if img_data:
                print(f"  got Wikipedia image for {name}", flush=True)
                if not is_face_present(img_data):
                    print(f"  no face in Wikipedia image for {name}", flush=True)
                    img_data = None
            else:
                print(f"  no Wikipedia image for {name}", flush=True)
        
        if not img_data:
            search_terms = [
                f"{name} headshot portrait",
                f"{name} photo",
                f"{name} face"
            ]
            for search_term in search_terms:
                if img_data:
                    break
                for attempt in range(2):
                    try:
                        print(f"  trying DDGS for {name} with '{search_term}' (attempt {attempt+1})...", flush=True)
                        time.sleep(3 + attempt * 3 + random.uniform(1, 3))
                        with DDGS() as ddgs:
                            results = list(ddgs.images(query=search_term, max_results=5))
                            print(f"  DDGS returned {len(results)} results for {name}", flush=True)
                            for r in results:
                                try:
                                    time.sleep(random.uniform(0.5, 1.5))
                                    resp = requests.get(r['image'], timeout=10, headers=headers)
                                    if resp.status_code == 200:
                                        print(f"  downloaded image from DDGS for {name}", flush=True)
                                        if is_face_present(resp.content):
                                            img_data = resp.content
                                            print(f"  face found in DDGS image for {name}", flush=True)
                                            break
                                        else:
                                            print(f"  no face in DDGS image for {name}", flush=True)
                                except Exception as e:
                                    print(f"  error downloading DDGS image: {e}", flush=True)
                                    continue
                        if img_data:
                            break
                    except Exception as e:
                        error_msg = str(e)
                        if "403" in error_msg or "Forbidden" in error_msg:
                            print(f"  DDGS rate limited for {name}, waiting longer...", flush=True)
                            time.sleep(5 + random.uniform(2, 5))
                        else:
                            print(f"  DDGS error for {name} (attempt {attempt+1}): {e}", flush=True)
                        if attempt < 1:
                            time.sleep(3)
        
        if img_data:
            if is_face_present(img_data):
                try:
                    pil_img = Image.open(io.BytesIO(img_data))
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    
                    img_array = np.array(pil_img, dtype=np.uint8)
                    img_array = np.ascontiguousarray(img_array, dtype=np.uint8)
                    
                    h, w = img_array.shape[:2]
                    if h > 2000 or w > 2000:
                        scale = min(2000.0 / h, 2000.0 / w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    filename = f"{clean_name}_{int(time.time())}.jpg"
                    filepath = os.path.join(celebs_dir, filename)
                    pil_img.save(filepath, 'JPEG', quality=95)
                    added += 1
                    print(f"added {name} ({added}/{len(celebs)})", flush=True)
                except Exception as e:
                    print(f"error saving {name}: {e}", flush=True)
                    traceback.print_exc()
                    continue
            else:
                print(f"  final check: no face detected in image for {name}", flush=True)
        else:
            print(f"  no image found for {name}", flush=True)
        
        time.sleep(2 + random.uniform(0.5, 1.5))
    
    print(f"done! added {added} celebrities")
    return added

if __name__ == "__main__":
    seed_celebs()
