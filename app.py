#!/usr/bin/env python3
# celebrity face matching thing

from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
try:
    import face_recognition
except ImportError as e:
    print(f"face_recognition import failed: {e}")
except Exception as e:
    print(f"face_recognition error: {e}")
import numpy as np
import os
import pickle
from PIL import Image
import base64
import io
import threading
import time
import requests
from ddgs import DDGS
import random
import json
import re

app = Flask(__name__)

matcher = None
_db_version = 0
_db_version_lock = threading.Lock()


def format_celebrity_name(filename_name):
    if not filename_name:
        return filename_name
    
    name = filename_name.rsplit('.', 1)[0]
    name = re.sub(r'_\d+$', '', name)
    name = name.replace('_', ' ')
    name = name.title()
    
    return name


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
    
    def load_database(self, force_rebuild=False):
        if os.path.exists(self.cache_file) and not force_rebuild:
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if data and len(data) > 0:
                        self.celeb_data = data
                        print(f"loaded {len(self.celeb_data)} celebs from cache", flush=True)
                        return self.celeb_data
            except Exception as e:
                print(f"cache error: {e}", flush=True)
        
        self.celeb_data = []
        
        preload_dir = "preload"
        all_image_files = []
        
        if os.path.exists(preload_dir):
            preload_files = os.listdir(preload_dir)
            preload_images = [f for f in preload_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in preload_images:
                all_image_files.append(os.path.join(preload_dir, img))
        
        if not os.path.exists(self.celebs_dir):
            os.makedirs(self.celebs_dir, exist_ok=True)
        else:
            files = os.listdir(self.celebs_dir)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in image_files:
                all_image_files.append(os.path.join(self.celebs_dir, img))
        
        if len(all_image_files) == 0:
            return self.celeb_data
        
        processed_count = 0
        encoded_count = 0
        failed_count = 0
        
        for filepath in all_image_files:
            filename = os.path.basename(filepath)
            processed_count += 1
            try:
                rgb = None
                try:
                    rgb = face_recognition.load_image_file(filepath)
                except Exception as e1:
                    try:
                        pil_img = Image.open(filepath)
                        if pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                        rgb = np.array(pil_img, dtype=np.uint8)
                    except Exception as e2:
                        failed_count += 1
                        continue
                
                if rgb is None or len(rgb.shape) != 3 or rgb.shape[2] != 3:
                    failed_count += 1
                    continue
                
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                
                h, w = rgb.shape[:2]
                if h > 800 or w > 800:
                    scale = min(800.0 / h, 800.0 / w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    rgb = cv2.resize(rgb, (new_w, new_h))
                    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                try:
                    face_encs = face_recognition.face_encodings(rgb, num_jitters=1)
                except Exception as e:
                    failed_count += 1
                    continue
                
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
                    encoded_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
        
        print(f"database loaded: {len(self.celeb_data)} celebrities", flush=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.celeb_data, f)
        
        return self.celeb_data
    
    def find_match(self, frame_rgb):
        if not self.celeb_data or len(self.celeb_data) == 0:
            return None, None, None
        
        if len(frame_rgb.shape) != 3 or frame_rgb.shape[2] != 3:
            return None, None, None
        
        h, w = frame_rgb.shape[:2]
        small_rgb = cv2.resize(frame_rgb, (int(w * 0.25), int(h * 0.25)))
        small_rgb = np.ascontiguousarray(small_rgb, dtype=np.uint8)
        
        if small_rgb.dtype != np.uint8:
            small_rgb = small_rgb.astype(np.uint8)
        
        if small_rgb.min() < 0 or small_rgb.max() > 255:
            small_rgb = np.clip(small_rgb, 0, 255).astype(np.uint8)
        
        try:
            face_encs = face_recognition.face_encodings(small_rgb, num_jitters=1)
        except Exception as e:
            return None, None, None
        
        if not face_encs or len(face_encs) == 0:
            return None, None, None
        
        celeb_encs = [c['enc'] for c in self.celeb_data]
        dists = face_recognition.face_distance(celeb_encs, face_encs[0])
        idx = np.argmin(dists)
        distance = dists[idx]
        similarity = max(0, (1 - distance) * 100)
        
        return self.celeb_data[idx], distance, similarity


try:
    matcher = CelebrityMatcher()
    matcher.load_database()
    celeb_count = len(matcher.celeb_data)
    
    if celeb_count == 0:
        def seed_background():
            try:
                from seed_celebs import seed_celebs
                added = seed_celebs()
                if matcher:
                    matcher.load_database()
            except Exception as e:
                pass
        
        threading.Thread(target=seed_background, daemon=True).start()
except Exception as e:
    matcher = None


def calculate_face_statistics(user_lms, match_lms):
    if user_lms is None or match_lms is None:
        return None
    
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    NOSE_INDICES = [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 290, 305, 290, 305]
    MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
    FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136]
    
    stats = {
        'overall_similarity': 0.0,
        'angles': {},
        'regions': {}
    }
    
    errors = np.linalg.norm(user_lms - match_lms, axis=1)
    avg_error = np.mean(errors)
    stats['overall_similarity'] = max(0, (1 - avg_error * 10) * 100)
    
    nose_tip_idx = 1
    chin_idx = 175
    if len(user_lms) > chin_idx and len(match_lms) > chin_idx:
        user_face_vec = user_lms[nose_tip_idx] - user_lms[chin_idx]
        match_face_vec = match_lms[nose_tip_idx] - match_lms[chin_idx]
        user_angle = np.arctan2(user_face_vec[1], user_face_vec[0]) * 180 / np.pi
        match_angle = np.arctan2(match_face_vec[1], match_face_vec[0]) * 180 / np.pi
        angle_diff = abs(user_angle - match_angle)
        stats['angles']['face_angle'] = {
            'user': round(user_angle, 1),
            'match': round(match_angle, 1),
            'difference': round(angle_diff, 1)
        }
    
    def calculate_region_similarity(indices):
        if not indices:
            return 0.0
        valid_indices = [i for i in indices if i < len(user_lms) and i < len(match_lms)]
        if not valid_indices:
            return 0.0
        region_errors = errors[valid_indices]
        avg_error = np.mean(region_errors)
        similarity = max(0, (1 - avg_error * 10) * 100)
        return round(similarity, 1)
    
    stats['regions'] = {
        'left_eye': calculate_region_similarity(LEFT_EYE_INDICES),
        'right_eye': calculate_region_similarity(RIGHT_EYE_INDICES),
        'nose': calculate_region_similarity(NOSE_INDICES),
        'mouth': calculate_region_similarity(MOUTH_INDICES),
        'face_oval': calculate_region_similarity(FACE_OVAL_INDICES)
    }
    
    most_similar = max(stats['regions'].items(), key=lambda x: x[1])
    stats['most_similar_region'] = {
        'name': most_similar[0].replace('_', ' ').title(),
        'similarity': most_similar[1]
    }
    
    least_similar = min(stats['regions'].items(), key=lambda x: x[1])
    stats['least_similar_region'] = {
        'name': least_similar[0].replace('_', ' ').title(),
        'similarity': least_similar[1]
    }
    
    return stats


def draw_wireframe_on_image(img_array, match_data=None):
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return Image.fromarray(img_array)
    
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img = np.ascontiguousarray(rgb_img, dtype=np.uint8)
    
    if rgb_img.dtype != np.uint8:
        rgb_img = rgb_img.astype(np.uint8)
    
    if rgb_img.min() < 0 or rgb_img.max() > 255:
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True, 
        max_num_faces=1
    )
    
    results = face_mesh.process(rgb_img)
    
    if results.multi_face_landmarks:
        lm_list = results.multi_face_landmarks[0].landmark
        connections = mp_face_mesh.FACEMESH_TESSELATION
        
        if match_data and 'lms' in match_data:
            curr_lms = np.array([(l.x, l.y) for l in lm_list])
            curr_norm = curr_lms - curr_lms.mean(axis=0)
            errors = np.linalg.norm(curr_norm - match_data['lms'], axis=1)
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
        else:
            for conn in connections:
                x1 = int(lm_list[conn[0]].x * w)
                y1 = int(lm_list[conn[0]].y * h)
                x2 = int(lm_list[conn[1]].x * w)
                y2 = int(lm_list[conn[1]].y * h)
                cv2.line(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def process_frame(img_array, matcher):
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return None, None, None
    
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img = np.ascontiguousarray(rgb_img, dtype=np.uint8)
    
    if rgb_img.dtype != np.uint8:
        rgb_img = rgb_img.astype(np.uint8)
    
    if rgb_img.min() < 0 or rgb_img.max() > 255:
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True, 
        max_num_faces=1
    )
    
    results = face_mesh.process(rgb_img)
    
    match = None
    similarity = 0.0
    statistics = None
    
    if results.multi_face_landmarks:
        user_lm_list = results.multi_face_landmarks[0].landmark
        user_lms = np.array([(l.x, l.y) for l in user_lm_list])
        user_lms_norm = user_lms - user_lms.mean(axis=0)
        
        if matcher and matcher.celeb_data and len(matcher.celeb_data) > 0:
            match, distance, similarity = matcher.find_match(rgb_img)
            if match and 'lms' in match and match['lms'] is not None:
                statistics = calculate_face_statistics(user_lms_norm, match['lms'])
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_img_with_wireframe = draw_wireframe_on_image(img_rgb, match)
    
    return processed_img_with_wireframe, match, similarity, statistics


def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64.decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    celeb_count = len(matcher.celeb_data) if matcher else 0
    celebs_dir = "celebs"
    file_count = 0
    if os.path.exists(celebs_dir):
        file_count = len([f for f in os.listdir(celebs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    return jsonify({
        'celeb_count': celeb_count,
        'file_count': file_count,
        'matcher_initialized': matcher is not None
    })


@app.route('/api/reload')
def reload_database():
    if matcher:
        if os.path.exists(matcher.cache_file):
            os.remove(matcher.cache_file)
        matcher.load_database(force_rebuild=True)
        return jsonify({
            'success': True,
            'count': len(matcher.celeb_data),
            'message': f'reloaded {len(matcher.celeb_data)} celebrities'
        })
    return jsonify({'error': 'matcher not initialized'}), 500


@app.route('/api/seed', methods=['POST'])
def trigger_seed():
    if not matcher:
        return jsonify({'error': 'matcher not initialized'}), 500
    
    def seed_background():
        try:
            from seed_celebs import seed_celebs
            added = seed_celebs()
            if matcher:
                if os.path.exists(matcher.cache_file):
                    os.remove(matcher.cache_file)
                matcher.load_database(force_rebuild=True)
        except Exception as e:
            pass
    
    threading.Thread(target=seed_background, daemon=True).start()
    return jsonify({'success': True, 'message': 'seeding started in background'})


@app.route('/api/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'no image selected'}), 400
        
        image = Image.open(io.BytesIO(file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image, dtype=np.uint8)
        
        if not matcher:
            return jsonify({'error': 'matcher not initialized'}), 500
        
        if matcher.celeb_data and len(matcher.celeb_data) == 0:
            if os.path.exists(matcher.cache_file):
                os.remove(matcher.cache_file)
            matcher.load_database(force_rebuild=True)
        
        try:
            processed_img, match, similarity, statistics = process_frame(image_array, matcher)
            if processed_img is None:
                return jsonify({'error': 'failed to process image'}), 500
        except Exception as e:
            return jsonify({'error': f'error processing image: {str(e)}'}), 500
        
        if not match:
            try:
                processed_img_str = image_to_base64(processed_img)
                celeb_count = len(matcher.celeb_data) if matcher else 0
                match_name = 'No match found' if celeb_count > 0 else 'No celebrities yet'
                return jsonify({
                    'success': True,
                    'match_name': match_name,
                    'similarity': 0.0,
                    'processed_image': processed_img_str,
                    'celebrity_image': processed_img_str,
                    'face_count': celeb_count
                })
            except Exception as e:
                print(f"error encoding image: {e}")
                return jsonify({'error': 'error encoding image'}), 500
        
        try:
            processed_img_str = image_to_base64(processed_img)
            
            celeb_img = Image.open(match['img_path'])
            if celeb_img.mode != 'RGB':
                celeb_img = celeb_img.convert('RGB')
            celeb_img_array = np.array(celeb_img, dtype=np.uint8)
            celeb_img_with_wireframe = draw_wireframe_on_image(celeb_img_array, match)
            celeb_img_str = image_to_base64(celeb_img_with_wireframe)
            
            display_name = format_celebrity_name(match['name'])
            
            response_data = {
                'success': True,
                'match_name': display_name,
                'similarity': float(similarity),
                'processed_image': processed_img_str,
                'celebrity_image': celeb_img_str,
                'face_count': len(matcher.celeb_data) if matcher else 0
            }
            
            if statistics:
                response_data['statistics'] = statistics
            
            return jsonify(response_data)
        except Exception as e:
            return jsonify({'error': f'error preparing response: {str(e)}'}), 500
        
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
        
        celebs_dir = "celebs"
        if not os.path.exists(celebs_dir):
            os.makedirs(celebs_dir, exist_ok=True)
        
        filename = f"{name.lower().replace(' ', '_')}_{int(time.time())}.jpg"
        filepath = os.path.join(celebs_dir, filename)
        
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            h, w = pil_img.size[1], pil_img.size[0]
            if h > 800 or w > 800:
                scale = min(800.0 / h, 800.0 / w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            pil_img.save(filepath, 'JPEG', quality=95)
            time.sleep(0.1)
            if not os.path.exists(filepath):
                return jsonify({'error': 'failed to save image file'}), 500
            
            rgb = face_recognition.load_image_file(filepath)
            
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            
            rgb = rgb.copy()
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            
            if len(rgb.shape) != 3 or rgb.shape[2] != 3:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': 'invalid image format'}), 400
            
            
            face_encs = face_recognition.face_encodings(rgb, num_jitters=1)
            
            if not face_encs or len(face_encs) == 0:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': 'no face detected in image'}), 400
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'face encoding failed: {str(e)}'}), 400
        
        if matcher:
            old_count = len(matcher.celeb_data)
            cache_path = matcher.cache_file
            if os.path.exists(cache_path):
                os.remove(cache_path)
            
            matcher.celeb_data = []
            matcher.load_database(force_rebuild=True)
        
        return jsonify({
            'success': True, 
            'message': f'registered {name} successfully!',
            'count': len(matcher.celeb_data) if matcher else 0
        })
        
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
        TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
        if not TMDB_API_KEY:
            return None
        url = f"https://api.themoviedb.org/3/search/person"
        params = {"api_key": TMDB_API_KEY, "query": name}
        data = requests.get(url, params=params, timeout=3).json()
        if data.get('results'):
            path = data['results'][0].get('profile_path')
            if path:
                img_url = f"https://image.tmdb.org/t/p/h632{path}"
                resp = requests.get(img_url, timeout=5)
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
        resp = requests.get(url, params=params, timeout=3).json()
        pages = resp.get("query", {}).get("pages", {})
        for p in pages.values():
            if "thumbnail" in p:
                img_resp = requests.get(p["thumbnail"]["source"], timeout=5)
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
            if matcher:
                if os.path.exists(matcher.cache_file):
                    os.remove(matcher.cache_file)
                matcher.load_database(force_rebuild=True)
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
        if not img_data:
            img_data = get_wikipedia_image(name)
        
        if not img_data:
            for attempt in range(2):
                try:
                    if attempt > 0:
                        time.sleep(0.5)
                    with DDGS() as ddgs:
                        results = list(ddgs.images(query=f"{name} headshot portrait", max_results=8))
                        for r in results[:5]:
                            try:
                                resp = requests.get(r['image'], timeout=5, headers=headers)
                                if resp.status_code == 200:
                                    img_data = resp.content
                                    break
                            except:
                                continue
                        if img_data:
                            break
                except Exception as e:
                    if attempt < 1:
                        time.sleep(0.5)
        
        if img_data and is_face_present(img_data):
            filename = f"{clean_name}_{int(time.time())}.jpg"
            filepath = os.path.join(celebs_dir, filename)
            
            try:
                pil_img = Image.open(io.BytesIO(img_data))
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                rgb = np.array(pil_img, dtype=np.uint8)
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                
                h, w = rgb.shape[:2]
                if h > 800 or w > 800:
                    scale = min(800.0 / h, 800.0 / w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    rgb = cv2.resize(rgb, (new_w, new_h))
                    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                
                face_encs = face_recognition.face_encodings(rgb, num_jitters=1)
                if not face_encs or len(face_encs) == 0:
                    return jsonify({
                        'error': f'no face detected in image for {name}'
                    }), 400
                
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, img_bgr)
            except Exception as e:
                return jsonify({
                    'error': f'failed to process image for {name}: {str(e)}'
                }), 400
            
            if matcher:
                if os.path.exists(matcher.cache_file):
                    os.remove(matcher.cache_file)
                matcher.load_database(force_rebuild=True)
            
            return jsonify({
                'success': True,
                'message': f'added {name} to database!',
                'count': len(matcher.celeb_data) if matcher else 0
            })
        else:
            return jsonify({
                'error': f'couldnt find a face image for {name}, try different name'
            }), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'no file provided'}), 400
        
        file = request.files['file']
        name = request.form.get('name', '').strip()
        
        if file.filename == '':
            return jsonify({'error': 'no file selected'}), 400
        
        if not name:
            name = os.path.splitext(file.filename)[0]
        
        celebs_dir = "celebs"
        if not os.path.exists(celebs_dir):
            os.makedirs(celebs_dir, exist_ok=True)
        
        filename = f"{name.lower().replace(' ', '_')}_{int(time.time())}.jpg"
        filepath = os.path.join(celebs_dir, filename)
        
        try:
            pil_img = Image.open(file)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            rgb = np.array(pil_img, dtype=np.uint8)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            
            h, w = rgb.shape[:2]
            if h > 2000 or w > 2000:
                scale = min(2000.0 / h, 2000.0 / w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                rgb = cv2.resize(rgb, (new_w, new_h))
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            
            face_encs = face_recognition.face_encodings(rgb)
            if not face_encs or len(face_encs) == 0:
                return jsonify({'error': 'no face detected in image'}), 400
            
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, img_bgr)
        except Exception as e:
            return jsonify({'error': f'failed to process image: {str(e)}'}), 400
        
        if matcher:
            if os.path.exists(matcher.cache_file):
                os.remove(matcher.cache_file)
            matcher.load_database(force_rebuild=True)
        
        return jsonify({
            'success': True,
            'message': f'uploaded and registered {name} successfully!',
            'count': len(matcher.celeb_data) if matcher else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_bulk', methods=['POST'])
def upload_bulk():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'no files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': 'no files selected'}), 400
        
        celebs_dir = "celebs"
        if not os.path.exists(celebs_dir):
            os.makedirs(celebs_dir, exist_ok=True)
        
        added = 0
        failed = 0
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            name = os.path.splitext(file.filename)[0]
            filename = f"{name.lower().replace(' ', '_')}_{int(time.time())}.jpg"
            filepath = os.path.join(celebs_dir, filename)
            
            try:
                pil_img = Image.open(file)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                h, w = pil_img.size[1], pil_img.size[0]
                if h > 800 or w > 800:
                    scale = min(800.0 / h, 800.0 / w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                pil_img.save(filepath, 'JPEG', quality=95)
                
                rgb = face_recognition.load_image_file(filepath)
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                
                face_encs = face_recognition.face_encodings(rgb, num_jitters=1)
                if not face_encs or len(face_encs) == 0:
                    os.remove(filepath)
                    failed += 1
                    results.append({'file': file.filename, 'status': 'failed', 'reason': 'no face detected'})
                    continue
                
                added += 1
                results.append({'file': file.filename, 'status': 'success', 'name': name})
                
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                failed += 1
                results.append({'file': file.filename, 'status': 'failed', 'reason': str(e)})
        
        if matcher and added > 0:
            if os.path.exists(matcher.cache_file):
                os.remove(matcher.cache_file)
            matcher.load_database(force_rebuild=True)
        
        return jsonify({
            'success': True,
            'message': f'uploaded {added} images successfully, {failed} failed',
            'added': added,
            'failed': failed,
            'results': results,
            'total_count': len(matcher.celeb_data) if matcher else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production' and not os.environ.get('RAILWAY_ENVIRONMENT')
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
