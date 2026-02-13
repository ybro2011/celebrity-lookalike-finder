#!/usr/bin/env python3

import streamlit as st
import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import time
import pickle
from PIL import Image
import threading

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
                    self.celeb_data = pickle.load(f)
                return self.celeb_data
            except Exception as e:
                st.warning(f"error loading cache: {e}")
        
        self.celeb_data = []
        
        if not os.path.exists(self.celebs_dir):
            os.makedirs(self.celebs_dir, exist_ok=True)
            return self.celeb_data
        
        for filename in os.listdir(self.celebs_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.celebs_dir, filename)
                try:
                    img_bgr = cv2.imread(filepath)
                    if img_bgr is None:
                        continue
                    
                    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    face_encs = face_recognition.face_encodings(rgb)
                    
                    if face_encs:
                        name = os.path.splitext(filename)[0]
                        lms = self.get_norm_lms(img_bgr)
                        
                        self.celeb_data.append({
                            'name': name,
                            'enc': face_encs[0],
                            'lms': lms if lms is not None else np.zeros((468, 2)),
                            'img_path': filepath
                        })
                except Exception as e:
                    continue
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.celeb_data, f)
        
        return self.celeb_data
    
    def find_match(self, frame_rgb):
        if not self.celeb_data:
            return None, None, None
        
        small_rgb = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
        face_encs = face_recognition.face_encodings(small_rgb)
        
        if not face_encs:
            return None, None, None
        
        dists = face_recognition.face_distance(
            [c['enc'] for c in self.celeb_data], 
            face_encs[0]
        )
        idx = np.argmin(dists)
        distance = dists[idx]
        similarity = max(0, (1 - distance) * 100)
        
        return self.celeb_data[idx], distance, similarity


_db_version_lock = threading.Lock()
_db_version = 0

def get_db_version():
    global _db_version
    with _db_version_lock:
        return _db_version

def increment_db_version():
    global _db_version
    with _db_version_lock:
        _db_version += 1

@st.cache_resource
def load_celebrity_data(_version):
    matcher = CelebrityMatcher()
    celeb_data = matcher.load_database()
    return matcher, celeb_data


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
                avg_err = (errors[conn[0]] + errors[conn[1]]) / 2
                g = max(0, 255 - int(avg_err * 8500))
                r = min(255, int(avg_err * 8500))
                
                pt1 = (int(lm_list[conn[0]].x * w), int(lm_list[conn[0]].y * h))
                pt2 = (int(lm_list[conn[1]].x * w), int(lm_list[conn[1]].y * h))
                cv2.line(img, pt1, pt2, (0, g, r), 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), match, similarity


def register_new_face(name, image_array):
    if not name.strip():
        return False
    
    celebs_dir = "celebs"
    if not os.path.exists(celebs_dir):
        os.makedirs(celebs_dir, exist_ok=True)
    
    filename = f"{name.lower().replace(' ', '_')}_{int(time.time())}.jpg"
    filepath = os.path.join(celebs_dir, filename)
    
    cv2.imwrite(filepath, image_array)
    
    increment_db_version()
    return True


def main():
    st.set_page_config(
        page_title="How much do you look like a celebrity?",
        layout="centered"
    )
    
    st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a1a;
    }
    
    .main .block-container {
        background-color: #1a1a1a;
    }
    
    header[data-testid="stHeader"] {
        background-color: #1a1a1a;
        border-bottom: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    .stDecoration {
        display: none;
    }
    
    div[data-testid="stCameraInput"] {
        max-width: 600px;
        margin: 0 auto;
        position: relative;
        padding: 60px;
    }
    
    div[data-testid="stCameraInput"] > div {
        border: 40px solid #8B0000;
        border-radius: 4px;
        box-shadow: 0 0 20px rgba(139, 0, 0, 0.6);
        position: relative;
    }
    
    .movie-lightbulb {
        position: absolute;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #FFD700;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.8), 0 0 20px rgba(255, 215, 0, 0.6);
        z-index: 100;
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'processed_img' not in st.session_state:
        st.session_state.processed_img = None
    if 'match' not in st.session_state:
        st.session_state.match = None
    if 'similarity' not in st.session_state:
        st.session_state.similarity = 0.0
    if 'image_array' not in st.session_state:
        st.session_state.image_array = None
    
    try:
        with st.spinner("Loading celebrity database..."):
            matcher, celeb_data = load_celebrity_data(get_db_version())
        if len(celeb_data) == 0:
            st.warning("No celebrities found in the database.")
    except Exception as e:
        st.error(f"Error loading celebrity database: {str(e)}")
        st.stop()
        return
    
    if not st.session_state.show_results:
        st.markdown("""
        <script>
        (function() {
            function createLightbulb(x, y) {
                const bulb = document.createElement('div');
                bulb.className = 'movie-lightbulb';
                bulb.style.position = 'absolute';
                bulb.style.left = x + 'px';
                bulb.style.top = y + 'px';
                bulb.style.width = '14px';
                bulb.style.height = '14px';
                bulb.style.borderRadius = '50%';
                bulb.style.background = '#FFD700';
                bulb.style.boxShadow = '0 0 10px rgba(255, 215, 0, 0.8), 0 0 20px rgba(255, 215, 0, 0.6)';
                bulb.style.zIndex = '100';
                bulb.style.pointerEvents = 'none';
                return bulb;
            }
            
            function addLightbulbs() {
                const container = document.querySelector('div[data-testid="stCameraInput"]');
                if (!container) {
                    setTimeout(addLightbulbs, 200);
                    return;
                }
                
                if (container.dataset.bulbsAdded === 'true') return;
                
                const cameraDiv = container.querySelector('div');
                if (!cameraDiv || cameraDiv.offsetWidth === 0) {
                    setTimeout(addLightbulbs, 200);
                    return;
                }
                
                container.dataset.bulbsAdded = 'true';
                
                const offset = 52;
                const numPerSide = 5;
                const width = cameraDiv.offsetWidth;
                const height = cameraDiv.offsetHeight;
                const divOffsetLeft = cameraDiv.offsetLeft;
                const divOffsetTop = cameraDiv.offsetTop;
                
                for (let i = 0; i < numPerSide; i++) {
                    const x = divOffsetLeft + (width / (numPerSide - 1)) * i - 7;
                    const y = divOffsetTop - offset - 7;
                    container.appendChild(createLightbulb(x, y));
                }
                
                for (let i = 1; i < numPerSide - 1; i++) {
                    const x = divOffsetLeft + width + offset - 7;
                    const y = divOffsetTop + (height / (numPerSide - 1)) * i - 7;
                    container.appendChild(createLightbulb(x, y));
                }
                
                for (let i = numPerSide - 1; i >= 0; i--) {
                    const x = divOffsetLeft + (width / (numPerSide - 1)) * i - 7;
                    const y = divOffsetTop + height + offset - 7;
                    container.appendChild(createLightbulb(x, y));
                }
                
                for (let i = numPerSide - 2; i > 0; i--) {
                    const x = divOffsetLeft - offset - 7;
                    const y = divOffsetTop + (height / (numPerSide - 1)) * i - 7;
                    container.appendChild(createLightbulb(x, y));
                }
            }
            
            setTimeout(addLightbulbs, 300);
            setTimeout(addLightbulbs, 800);
            setTimeout(addLightbulbs, 1500);
            
            const observer = new MutationObserver(() => {
                const container = document.querySelector('div[data-testid="stCameraInput"]');
                if (container && container.dataset.bulbsAdded !== 'true') {
                    setTimeout(addLightbulbs, 100);
                }
            });
            
            observer.observe(document.body, { childList: true, subtree: true });
        })();
        </script>
        """, unsafe_allow_html=True)
        
        camera_file = st.camera_input("", label_visibility="hidden")
        
        if camera_file is not None:
            image = Image.open(camera_file)
            image_array = np.array(image.convert('RGB'))
            
            with st.spinner("Finding your celebrity match..."):
                processed_img, match, similarity = process_frame(image_array, matcher)
            
            if match:
                st.session_state.processed_img = processed_img
                st.session_state.match = match
                st.session_state.similarity = similarity
                st.session_state.image_array = image_array
                st.session_state.show_results = True
                st.rerun()
            else:
                st.warning("No face detected in the image. Please try again.")
    
    else:
        if st.button("← Take Another Photo"):
            st.session_state.show_results = False
            st.session_state.processed_img = None
            st.session_state.match = None
            st.session_state.similarity = 0.0
            st.session_state.image_array = None
            st.rerun()
        
        if st.session_state.match:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(st.session_state.processed_img, caption="Your photo with face mesh", use_container_width=True)
            
            with col2:
                st.subheader(f"Match: {st.session_state.match['name']}")
                st.metric("Similarity", f"{st.session_state.similarity:.1f}%")
                
                if 'img_path' in st.session_state.match and os.path.exists(st.session_state.match['img_path']):
                    celeb_img = Image.open(st.session_state.match['img_path'])
                    st.image(celeb_img, caption=st.session_state.match['name'], use_container_width=True)
            
            st.divider()
            st.subheader("Register Your Face?")
            with st.form("register_form", clear_on_submit=True):
                name = st.text_input("Enter your name:", placeholder="Your name here")
                col_submit, col_cancel = st.columns(2)
                
                with col_submit:
                    submitted = st.form_submit_button("Register", use_container_width=True)
                
                if submitted and name.strip():
                    img_bgr = cv2.cvtColor(st.session_state.image_array, cv2.COLOR_RGB2BGR)
                    if register_new_face(name, img_bgr):
                        st.success(f"Registered {name} successfully!")
                        st.cache_resource.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to register. Please try again.")
                elif submitted:
                    st.warning("Please enter a name.")


if __name__ == "__main__":
    main()
