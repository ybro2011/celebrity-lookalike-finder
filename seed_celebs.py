#!/usr/bin/env python3
import os
import requests
import cv2
import mediapipe as mp
import numpy as np
import time
from duckduckgo_search import DDGS

TMDB_API_KEY = "928d45ccb3dae4dcce45dbba02d64ca2"

def is_face_present(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return False
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.6
        )
        results = face_detector.process(rgb_img)
        return results.detections is not None
    except:
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
    
    # popular celebs to seed
    celebs = [
        "Taylor Swift", "Ariana Grande", "Billie Eilish", "Justin Bieber", 
        "Dua Lipa", "Rihanna", "Beyonce", "Lady Gaga", "Drake", "The Weeknd",
        "Harry Styles", "Olivia Rodrigo", "Selena Gomez", "Ed Sheeran", "Adele",
        "Bruno Mars", "Zendaya", "Tom Holland", "Robert Downey Jr", "Scarlett Johansson",
        "Dwayne Johnson", "Tom Cruise", "Brad Pitt", "Leonardo DiCaprio", "Margot Robbie",
        "Jennifer Lawrence", "Ryan Reynolds", "Timothee Chalamet", "Jenna Ortega",
        "Cillian Murphy", "Pedro Pascal", "Keanu Reeves", "Will Smith", "Angelina Jolie",
        "Jennifer Lopez", "Gal Gadot", "Chris Evans", "Chris Hemsworth", "Johnny Depp",
        "Emma Watson", "Sydney Sweeney", "Millie Bobby Brown", "Ryan Gosling",
        "Ben Affleck", "Jennifer Aniston", "Adam Sandler", "Cristiano Ronaldo",
        "Lionel Messi", "LeBron James", "Stephen Curry", "Serena Williams",
        "Elon Musk", "Jeff Bezos", "Bill Gates", "Mark Zuckerberg", "Oprah Winfrey",
        "Kim Kardashian", "Kylie Jenner", "Kendall Jenner", "Gigi Hadid", "Bella Hadid",
        "Shakira", "Nicki Minaj", "Cardi B", "Bad Bunny", "Travis Scott", "Zayn Malik",
        "Doja Cat", "Blake Lively", "Austin Butler", "Jacob Elordi", "Caitlin Clark",
        "Lewis Hamilton", "Max Verstappen", "Novak Djokovic", "Tiger Woods", "Virat Kohli",
        "Neymar Jr", "Kylian Mbappe", "Shohei Ohtani", "Patrick Mahomes", "Simone Biles",
        "Charles Leclerc", "Roger Federer", "Kevin Durant", "Giannis Antetokounmpo",
        "Sam Altman", "Lisa Su", "Jensen Huang", "Joe Rogan",
        "Sabrina Carpenter", "Olivia Wilde", "Florence Pugh", "Anya Taylor-Joy"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    added = 0
    print("seeding celebrities...")
    
    for name in celebs:
        clean_name = name.replace(" ", "_").lower()
        
        # check if already exists
        existing = [f for f in os.listdir(celebs_dir) if f.lower().startswith(clean_name)]
        if existing:
            continue
        
        img_data = None
        
        # try tmdb
        img_data = get_tmdb_image(name)
        if img_data and not is_face_present(img_data):
            img_data = None
        
        # try wikipedia
        if not img_data:
            img_data = get_wikipedia_image(name)
            if img_data and not is_face_present(img_data):
                img_data = None
        
        # try duckduckgo
        if not img_data:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(keywords=f"{name} headshot portrait", max_results=2))
                    for r in results:
                        resp = requests.get(r['image'], timeout=5, headers=headers)
                        if resp.status_code == 200 and is_face_present(resp.content):
                            img_data = resp.content
                            break
            except:
                pass
        
        if img_data and is_face_present(img_data):
            filename = f"{clean_name}_{int(time.time())}.jpg"
            filepath = os.path.join(celebs_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(img_data)
            added += 1
            print(f"added {name} ({added}/{len(celebs)})")
        
        time.sleep(0.5)  # be nice to APIs
    
    print(f"done! added {added} celebrities")
    return added

if __name__ == "__main__":
    seed_celebs()
