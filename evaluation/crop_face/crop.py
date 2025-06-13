import dlib
import os
import numpy as np
from PIL import Image
import torch
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

class EfficientFaceDetector:
    def __init__(
        self,
        cnn_model_path: str = 'mmod_human_face_detector.dat',
        landmark_model_path: str = 'shape_predictor_5_face_landmarks.dat',
        batch_size: int = 32,
        max_size: int = 800,
        num_workers: int = 4
    ):
        self.detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
        self.predictor = dlib.shape_predictor(landmark_model_path)
        
        self.batch_size = batch_size
        self.max_size = max_size
        self.num_workers = num_workers
        
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print("Using CUDA for face detection")
    
    def preprocess_image(self, image_path: str) -> tuple:
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_array = np.array(img)
        
        h, w = img_array.shape[:2]
        scale = 1.0
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_array = dlib.resize_image(img_array, rows=new_h, cols=new_w)
            
        return img_array, scale, (h, w)
    
    def process_batch(
        self, 
        batch_paths: List[str],
        output_dir: str,
        size: int = 300,
        padding: float = 0.25
    ) -> List[Dict]:
        results = []
        
        processed_images = []
        scales = []
        original_sizes = []
        
        for img_path in batch_paths:
            try:
                img_array, scale, orig_size = self.preprocess_image(img_path)
                processed_images.append(img_array)
                scales.append(scale)
                original_sizes.append(orig_size)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        batch_dets = []
        for img in processed_images:
            dets = self.detector(img, 1)
            batch_dets.append(dets)
        
        for idx, (img_path, img_array, dets, scale, orig_size) in enumerate(
            zip(batch_paths, processed_images, batch_dets, scales, original_sizes)
        ):
            faces = dlib.full_object_detections()
            
            for det_idx, detection in enumerate(dets):
                rect = detection.rect
                shape = self.predictor(img_array, rect)
                faces.append(shape)
                
                face_chip = dlib.get_face_chip(img_array, shape, size=size, padding=padding)
                
                output_name = f"{Path(img_path).stem}_face_{det_idx}.jpg"
                output_path = os.path.join(output_dir, output_name)
                Image.fromarray(face_chip).save(output_path, quality=95)
                
                if scale != 1.0:
                    bbox = [
                        int(rect.left() / scale),
                        int(rect.top() / scale),
                        int(rect.right() / scale),
                        int(rect.bottom() / scale)
                    ]
                else:
                    bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                
                results.append({
                    'image_path': img_path,
                    'face_idx': det_idx,
                    'confidence': detection.confidence,
                    'bbox': bbox,
                    'output_path': output_path
                })
        
        return results

    def detect_and_crop(
        self,
        image_paths: List[str],
        output_dir: str,
        size: int = 300,
        padding: float = 0.25
    ) -> List[Dict]:
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size)):
            batch_paths = image_paths[i:i + self.batch_size]
            results = self.process_batch(batch_paths, output_dir, size, padding)
            all_results.extend(results)
        
        return all_results

def process_images_parallel(
    image_paths: List[str],
    output_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> List[Dict]:
    detector = EfficientFaceDetector(
        batch_size=batch_size,
        num_workers=num_workers
    )
    return detector.detect_and_crop(image_paths, output_dir)

if __name__ == "__main__":
    import glob
    
    dlib_models_dir = "dlib_models"
    
    image_paths = glob.glob('samples/*.png')
    
    detector = EfficientFaceDetector(
        cnn_model_path=os.path.join(dlib_models_dir, 'mmod_human_face_detector.dat'),
        landmark_model_path=os.path.join(dlib_models_dir, 'shape_predictor_5_face_landmarks.dat'),
        batch_size=32,
        num_workers=4
    )
    
    results = detector.detect_and_crop(
        image_paths=image_paths,
        output_dir='cropped_faces'
    )
    
    print(f"Successfully processed {len(results)} faces")
    
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('detection_results.csv', index=False)