#!/usr/bin/env python3
# celebrity face matching thing

from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import pickle
from PIL import Image
import base64
import io
import threading
import time
import requests
from duckduckgo_search import DDGS
import random

app = Flask(__name__)

matcher = None
_db_version = 0
_db_version_lock = threading.Lock()


class CelebrityMatcher:
    def __init__(self, celebs_dir="celebs", cache_file="encodings.pickle"):
        self.celebs_dir = celebs_dir
        self.cache_file = cache_file
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            refine_landmarks=True, 
            max_num_faces=1
        )
        self.connections = self.mp_face_mesh.FACEMESH_TESSELATION
        self.celeb_data = []
    
    def get_norm_lms(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if res.multi_face_landmarks:
            coords = np.array([(l.x, l.y) for l in res.multi_face_landmarks[0].landmark])
            return coords - coords.mean(axis=0)
        return None
    
    def load_database(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if data and len(data) > 0:
                        self.celeb_data = data
                        return self.celeb_data
                    else:
                        print("cache file empty, rebuilding...")
            except Exception as e:
                print(f"error loading cache: {e}, rebuilding...")
        
        self.celeb_data = []
        
        if not os.path.exists(self.celebs_dir):
            os.makedirs(self.celebs_dir, exist_ok=True)
            return self.celeb_data
        
        files = os.listdir(self.celebs_dir)
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.celebs_dir, filename)
                try:
                    img_bgr = cv2.imread(filepath)
                    if img_bgr is None:
                        continue
                    
                    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    face_encs = face_recognition.face_encodings(rgb)
                    
                    if face_encs and len(face_encs) > 0:
                        name = os.path.splitext(filename)[0]
                        lms = self.get_norm_lms(img_bgr)
                        if lms is None:
                            lms = np.zeros((468, 2))
                        
                        self.celeb_data.append({
                            'name': name,
                            'enc': face_encs[0],
                            'lms': lms,
                            'img_path': filepath
                        })
                except Exception as e:
                    continue  # skip bad images
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.celeb_data, f)
        
        return self.celeb_data
    
    def find_match(self, frame_rgb):
        if not self.celeb_data or len(self.celeb_data) == 0:
            return None, None, None
        
        h, w = frame_rgb.shape[:2]
        new_w = int(w * 0.25)
        new_h = int(h * 0.25)
        small_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        
        face_encs = face_recognition.face_encodings(small_rgb)
        
        if not face_encs or len(face_encs) == 0:
            return None, None, None
        
        celeb_encs = [c['enc'] for c in self.celeb_data]
        dists = face_recognition.face_distance(celeb_encs, face_encs[0])
        idx = np.argmin(dists)
        distance = dists[idx]
        similarity = max(0, (1 - distance) * 100)
        
        return self.celeb_data[idx], distance, similarity


def process_frame(img_array, matcher):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True, 
        max_num_faces=1
    )
    
    results = face_mesh.process(rgb_img)
    
    match = None
    similarity = 0.0
    
    if results.multi_face_landmarks and matcher.celeb_data:
        match, distance, similarity = matcher.find_match(rgb_img)
        
        if match:
            lm_list = results.multi_face_landmarks[0].landmark
            curr_lms = np.array([(l.x, l.y) for l in lm_list])
            curr_norm = curr_lms - curr_lms.mean(axis=0)
            errors = np.linalg.norm(curr_norm - match['lms'], axis=1)
            connections = mp_face_mesh.FACEMESH_TESSELATION
            for conn in connections:
                err1 = errors[conn[0]]
                err2 = errors[conn[1]]
                avg_err = (err1 + err2) / 2
                g = max(0, 255 - int(avg_err * 8500))
                r = min(255, int(avg_err * 8500))
                
                x1 = int(lm_list[conn[0]].x * w)
                y1 = int(lm_list[conn[0]].y * h)
                x2 = int(lm_list[conn[1]].x * w)
                y2 = int(lm_list[conn[1]].y * h)
                cv2.line(img, (x1, y1), (x2, y2), (0, g, r), 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), match, similarity


def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64.decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'no image selected'}), 400
        
        image = Image.open(io.BytesIO(file.read()))
        image_array = np.array(image.convert('RGB'))
        
        processed_img, match, similarity = process_frame(image_array, matcher)
        
        if not match:
            return jsonify({'error': 'no face detected'}), 400
        
        processed_img_str = image_to_base64(processed_img)
        celeb_img = Image.open(match['img_path'])
        celeb_img_str = image_to_base64(celeb_img)
        
        return jsonify({
            'success': True,
            'match_name': match['name'],
            'similarity': float(similarity),
            'processed_image': processed_img_str,
            'celebrity_image': celeb_img_str,
            'face_count': len(matcher.celeb_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/register_face', methods=['POST'])
def register_face():
    try:
        data = request.json
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'error': 'name required'}), 400
        
        if 'image' not in data:
            return jsonify({'error': 'image required'}), 400
        
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image.convert('RGB'))
        
        celebs_dir = "celebs"
        if not os.path.exists(celebs_dir):
            os.makedirs(celebs_dir, exist_ok=True)
        
        filename = f"{name.lower().replace(' ', '_')}_{int(time.time())}.jpg"
        filepath = os.path.join(celebs_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        
        matcher.load_database()
        
        return jsonify({'success': True, 'message': f'registered {name} successfully!'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
            min_detection_confidence=0.6
        )
        results = face_detector.process(rgb_img)
        return results.detections is not None
    except:
        return False


def get_tmdb_image(name):
    try:
        TMDB_API_KEY = "928d45ccb3dae4dcce45dbba02d64ca2"
        url = f"https://api.themoviedb.org/3/search/person"
        params = {"api_key": TMDB_API_KEY, "query": name}
        data = requests.get(url, params=params, timeout=5).json()
        if data.get('results'):
            path = data['results'][0].get('profile_path')
            if path:
                img_url = f"https://image.tmdb.org/t/p/h632{path}"
                resp = requests.get(img_url, timeout=10)
                if resp.status_code == 200:
                    return resp.content
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
                if img_resp.status_code == 200:
                    return img_resp.content
        return None
    except:
        return None


@app.route('/api/suggest_celebrity', methods=['POST'])
def suggest_celebrity():
    try:
        data = request.json
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'error': 'celebrity name required'}), 400
        
        celebs_dir = "celebs"
        if not os.path.exists(celebs_dir):
            os.makedirs(celebs_dir, exist_ok=True)
        
        clean_name = name.replace(" ", "_").lower()
        
        existing_files = [f for f in os.listdir(celebs_dir) if f.lower().startswith(clean_name)]
        if existing_files:
            matcher.load_database()
            return jsonify({
                'success': True, 
                'message': f'{name} already exists!',
                'already_exists': True
            })
        
        img_data = None
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        img_data = get_tmdb_image(name)
        if img_data and not is_face_present(img_data):
            img_data = None
        
        if not img_data:
            img_data = get_wikipedia_image(name)
            if img_data and not is_face_present(img_data):
                img_data = None
        
        if not img_data:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(keywords=f"{name} headshot portrait", max_results=3))
                    for r in results:
                        resp = requests.get(r['image'], timeout=5, headers=headers)
                        if resp.status_code == 200 and is_face_present(resp.content):
                            img_data = resp.content
                            break
            except Exception as e:
                pass
        
        if img_data and is_face_present(img_data):
            filename = f"{clean_name}_{int(time.time())}.jpg"
            filepath = os.path.join(celebs_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            matcher.load_database()
            
            return jsonify({
                'success': True,
                'message': f'added {name} to database!'
            })
        else:
            return jsonify({
                'error': f'couldnt find a face image for {name}, try different name'
            }), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    matcher = CelebrityMatcher()
    matcher.load_database()
    print(f"loaded {len(matcher.celeb_data)} celebrities")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
