
#!/usr/bin/env python3
"""
NSFW Image Scanner Application
=============================

This application scans local folders for images containing nudity/NSFW content.
It uses state-of-the-art AI models running entirely locally on your PC.

Features:
- Local-only processing (no cloud uploads)
- Supports thousands of images and subfolders
- Multiple NSFW detection models
- GPU acceleration when available
- Detailed body part detection with confidence scores
- GUI interface and HTML report generation
- Cross-platform (Windows/Linux)

Author: AI Assistant
License: MIT
"""

import os
import sys
import json
import time
import logging
import threading
import webbrowser
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Image processing imports
import numpy as np
from PIL import Image, ImageTk
import cv2

# Model imports
try:
    import torch
    import torchvision.transforms as transforms
    from transformers import pipeline, AutoModelForImageClassification, ViTImageProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import opennsfw2 as n2
    OPENNSFW_AVAILABLE = True
except ImportError:
    OPENNSFW_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Structure to hold detection results for each image"""
    filename: str
    file_path: str
    file_size_bytes: int
    image_width: int
    image_height: int
    thumbnail_path: str

    # Detection results
    overall_nsfw_probability: float
    model_used: str

    # Category breakdowns (when available)
    categories: Dict[str, float]  # e.g., {"breasts": 0.85, "genitalia": 0.12}

    # Processing metadata
    processing_time: float
    error_message: Optional[str] = None


class NSFWModelManager:
    """
    Manages multiple NSFW detection models and automatically downloads them if needed.
    Prioritizes the best available model based on system capabilities.
    """

    def __init__(self):
        self.models = {}
        self.active_model = None
        self.device = "cuda" if torch.cuda.is_available() and TORCH_AVAILABLE else "cpu"
        self.model_cache_dir = Path.home() / ".nsfw_scanner" / "models"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

    def download_and_setup_models(self, progress_callback=None):
        """Download and initialize all available models"""
        logger.info("Setting up NSFW detection models...")

        # Try to load Falconsai model (best general performance)
        if TORCH_AVAILABLE:
            try:
                if progress_callback:
                    progress_callback("Downloading Falconsai NSFW model...")

                model = AutoModelForImageClassification.from_pretrained(
                    "Falconsai/nsfw_image_detection",
                    cache_dir=str(self.model_cache_dir)
                )
                processor = ViTImageProcessor.from_pretrained(
                    "Falconsai/nsfw_image_detection",
                    cache_dir=str(self.model_cache_dir)
                )

                model.to(self.device)
                model.eval()

                self.models["falconsai"] = {
                    "model": model,
                    "processor": processor,
                    "type": "binary"  # normal/nsfw classification
                }
                logger.info("Falconsai model loaded successfully")

            except Exception as e:
                logger.warning(f"Failed to load Falconsai model: {e}")

        # Try to load OpenNSFW2 model (fallback)
        if OPENNSFW_AVAILABLE:
            try:
                if progress_callback:
                    progress_callback("Setting up OpenNSFW2 model...")

                # OpenNSFW2 will auto-download weights if needed
                self.models["opennsfw2"] = {
                    "model": n2.make_open_nsfw_model(),
                    "type": "binary"
                }
                logger.info("OpenNSFW2 model loaded successfully")

            except Exception as e:
                logger.warning(f"Failed to load OpenNSFW2 model: {e}")

        # Try to load Marqo model (lightweight, high accuracy)
        if TORCH_AVAILABLE:
            try:
                if progress_callback:
                    progress_callback("Downloading Marqo NSFW model...")

                import timm
                model = timm.create_model(
                    "hf_hub:Marqo/nsfw-image-detection-384", 
                    pretrained=True
                )
                model.to(self.device)
                model.eval()

                data_config = timm.data.resolve_model_data_config(model)
                transforms_obj = timm.data.create_transform(**data_config, is_training=False)

                self.models["marqo"] = {
                    "model": model,
                    "transforms": transforms_obj,
                    "type": "binary"
                }
                logger.info("Marqo model loaded successfully")

            except Exception as e:
                logger.warning(f"Failed to load Marqo model: {e}")

        # Set the best available model as active
        if "marqo" in self.models:
            self.active_model = "marqo"
        elif "falconsai" in self.models:
            self.active_model = "falconsai"
        elif "opennsfw2" in self.models:
            self.active_model = "opennsfw2"
        else:
            raise RuntimeError("No NSFW detection models could be loaded!")

        logger.info(f"Using model: {self.active_model}")

    def predict_image(self, image_path: str, threshold: float = 0.5) -> Dict:
        """
        Predict NSFW content in an image using the active model

        Returns:
            Dict with prediction results including confidence scores
        """
        if not self.active_model:
            raise RuntimeError("No model is loaded!")

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            if self.active_model == "falconsai":
                return self._predict_falconsai(image, threshold)
            elif self.active_model == "marqo":
                return self._predict_marqo(image, threshold)
            elif self.active_model == "opennsfw2":
                return self._predict_opennsfw2(image, threshold)

        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {e}")
            return {
                "nsfw_probability": 0.0,
                "categories": {},
                "is_nsfw": False,
                "error": str(e)
            }

    def _predict_falconsai(self, image: Image.Image, threshold: float) -> Dict:
        """Predict using Falconsai model"""
        model_info = self.models["falconsai"]
        model = model_info["model"]
        processor = model_info["processor"]

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Assuming class 1 is NSFW
            nsfw_prob = probabilities[0][1].item()

            return {
                "nsfw_probability": nsfw_prob,
                "categories": {"general_nsfw": nsfw_prob},
                "is_nsfw": nsfw_prob > threshold,
                "model": "Falconsai ViT"
            }

    def _predict_marqo(self, image: Image.Image, threshold: float) -> Dict:
        """Predict using Marqo model"""
        model_info = self.models["marqo"]
        model = model_info["model"]
        transforms_obj = model_info["transforms"]

        with torch.no_grad():
            input_tensor = transforms_obj(image).unsqueeze(0).to(self.device)
            output = model(input_tensor).softmax(dim=-1).cpu()

            # Get class names from model config
            class_names = model.pretrained_cfg["label_names"]
            probabilities = output[0]

            # Find NSFW probability (assuming class 1 is NSFW)
            nsfw_prob = probabilities[1].item() if len(probabilities) > 1 else 0.0

            return {
                "nsfw_probability": nsfw_prob,
                "categories": {"general_nsfw": nsfw_prob},
                "is_nsfw": nsfw_prob > threshold,
                "model": "Marqo ViT-384"
            }

    def _predict_opennsfw2(self, image: Image.Image, threshold: float) -> Dict:
        """Predict using OpenNSFW2 model"""
        # Convert PIL to format expected by OpenNSFW2
        image_array = np.array(image)
        processed_image = n2.preprocess_image(image, n2.Preprocessing.YAHOO)

        model = self.models["opennsfw2"]["model"]
        inputs = np.expand_dims(processed_image, axis=0)
        predictions = model.predict(inputs)

        # predictions[0] = [sfw_probability, nsfw_probability]
        nsfw_prob = predictions[0][1]

        return {
            "nsfw_probability": nsfw_prob,
            "categories": {"general_nsfw": nsfw_prob},
            "is_nsfw": nsfw_prob > threshold,
            "model": "OpenNSFW2"
        }


class ImageScanner:
    """
    Main image scanning class that processes folders and generates reports
    """

    def __init__(self, model_manager: NSFWModelManager):
        self.model_manager = model_manager
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
        self.thumbnail_size = (128, 128)
        self.results = []

    def scan_folder(self, folder_path: str, threshold: float = 0.5, 
                   progress_callback=None, max_workers: int = 4) -> List[DetectionResult]:
        """
        Scan a folder and its subfolders for NSFW images

        Args:
            folder_path: Path to folder to scan
            threshold: NSFW confidence threshold (0.0-1.0)
            progress_callback: Function to call with progress updates
            max_workers: Number of parallel processing threads

        Returns:
            List of DetectionResult objects for flagged images
        """
        logger.info(f"Starting scan of folder: {folder_path}")

        # Find all image files
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if Path(file).suffix.lower() in self.supported_extensions:
                    image_files.append(os.path.join(root, file))

        logger.info(f"Found {len(image_files)} images to process")

        self.results = []
        processed_count = 0

        # Create thumbnails directory
        thumbnail_dir = Path(folder_path) / ".nsfw_thumbnails"
        thumbnail_dir.mkdir(exist_ok=True)

        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_image, img_path, threshold, thumbnail_dir): img_path
                for img_path in image_files
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    if result and result.overall_nsfw_probability > threshold:
                        self.results.append(result)

                    processed_count += 1

                    if progress_callback:
                        progress = (processed_count / len(image_files)) * 100
                        progress_callback(f"Processed {processed_count}/{len(image_files)} images ({progress:.1f}%)")

                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    processed_count += 1

        # Sort results by NSFW probability (highest first)
        self.results.sort(key=lambda x: x.overall_nsfw_probability, reverse=True)

        logger.info(f"Scan complete. Found {len(self.results)} flagged images")
        return self.results

    def _process_single_image(self, image_path: str, threshold: float, 
                            thumbnail_dir: Path) -> Optional[DetectionResult]:
        """Process a single image file"""
        try:
            start_time = time.time()

            # Get file info
            file_stat = os.stat(image_path)
            file_size = file_stat.st_size

            # Load image to get dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            # Run NSFW detection
            prediction = self.model_manager.predict_image(image_path, threshold)

            processing_time = time.time() - start_time

            # Only create thumbnail if image is flagged
            if prediction.get("is_nsfw", False):
                thumbnail_path = self._create_thumbnail(image_path, thumbnail_dir)
            else:
                return None  # Not flagged, don't include in results

            return DetectionResult(
                filename=os.path.basename(image_path),
                file_path=image_path,
                file_size_bytes=file_size,
                image_width=width,
                image_height=height,
                thumbnail_path=thumbnail_path,
                overall_nsfw_probability=prediction.get("nsfw_probability", 0.0),
                model_used=prediction.get("model", "Unknown"),
                categories=prediction.get("categories", {}),
                processing_time=processing_time,
                error_message=prediction.get("error")
            )

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return DetectionResult(
                filename=os.path.basename(image_path),
                file_path=image_path,
                file_size_bytes=0,
                image_width=0,
                image_height=0,
                thumbnail_path="",
                overall_nsfw_probability=0.0,
                model_used="Error",
                categories={},
                processing_time=0.0,
                error_message=str(e)
            )

    def _create_thumbnail(self, image_path: str, thumbnail_dir: Path) -> str:
        """Create a thumbnail for the image"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Create thumbnail
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)

                # Save thumbnail
                thumbnail_filename = f"thumb_{os.path.basename(image_path)}.jpg"
                thumbnail_path = thumbnail_dir / thumbnail_filename
                img.save(thumbnail_path, "JPEG", quality=85)

                return str(thumbnail_path)

        except Exception as e:
            logger.error(f"Failed to create thumbnail for {image_path}: {e}")
            return ""

    def generate_html_report(self, output_path: str) -> str:
        """Generate an HTML report of the scan results"""

        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSFW Image Scanner Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .summary {{
            background: #e8f4f8;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        .image-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: #fafafa;
        }}
        .image-card.high-confidence {{
            border-left: 4px solid #ff4444;
        }}
        .image-card.medium-confidence {{
            border-left: 4px solid #ffaa00;
        }}
        .image-card.low-confidence {{
            border-left: 4px solid #ffdd00;
        }}
        .thumbnail {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        .filename {{
            font-weight: bold;
            margin-bottom: 8px;
            word-break: break-all;
        }}
        .details {{
            font-size: 0.9em;
            color: #666;
        }}
        .confidence {{
            font-size: 1.1em;
            font-weight: bold;
            margin: 8px 0;
        }}
        .confidence.high {{ color: #d32f2f; }}
        .confidence.medium {{ color: #f57c00; }}
        .confidence.low {{ color: #fbc02d; }}
        .categories {{
            margin-top: 10px;
        }}
        .category {{
            display: inline-block;
            background: #e0e0e0;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin: 2px;
        }}
        .path {{
            font-family: monospace;
            background: #f0f0f0;
            padding: 4px 8px;
            border-radius: 4px;
            margin-top: 8px;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 NSFW Image Scanner Report</h1>

        <div class="summary">
            <h3>Scan Summary</h3>
            <p><strong>Images flagged:</strong> {total_flagged}</p>
            <p><strong>Scan completed:</strong> {scan_date}</p>
            <p><strong>Model used:</strong> {model_name}</p>
            <p><strong>Detection threshold:</strong> {threshold}%</p>
        </div>

        <div class="image-grid">
            {image_cards}
        </div>
    </div>
</body>
</html>"""

        # Generate image cards
        image_cards = []
        for result in self.results:
            confidence_class = self._get_confidence_class(result.overall_nsfw_probability)

            # Create categories display
            category_tags = []
            for cat, conf in result.categories.items():
                category_tags.append(f'<span class="category">{cat}: {conf:.1%}</span>')

            # Format file size
            file_size_str = self._format_file_size(result.file_size_bytes)

            card_html = f"""
            <div class="image-card {confidence_class}">
                <img src="{result.thumbnail_path}" alt="Thumbnail" class="thumbnail" onerror="this.style.display='none'">
                <div class="filename">{result.filename}</div>
                <div class="confidence {confidence_class}">
                    Confidence: {result.overall_nsfw_probability:.1%}
                </div>
                <div class="details">
                    <strong>Size:</strong> {file_size_str}<br>
                    <strong>Dimensions:</strong> {result.image_width}×{result.image_height}<br>
                    <strong>Processing time:</strong> {result.processing_time:.2f}s<br>
                    <strong>Model:</strong> {result.model_used}
                </div>
                <div class="categories">
                    {' '.join(category_tags)}
                </div>
                <div class="path">{result.file_path}</div>
            </div>
            """
            image_cards.append(card_html)

        # Fill in template
        model_name = self.results[0].model_used if self.results else "None"
        html_content = html_template.format(
            total_flagged=len(self.results),
            scan_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_name=model_name,
            threshold=50,  # Default threshold as percentage
            image_cards='\n'.join(image_cards)
        )

        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {output_path}")
        return output_path

    def _get_confidence_class(self, probability: float) -> str:
        """Get CSS class based on confidence level"""
        if probability >= 0.8:
            return "high-confidence"
        elif probability >= 0.6:
            return "medium-confidence"
        else:
            return "low-confidence"

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"


class NSFWScannerGUI:
    """
    Main GUI application for the NSFW scanner
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("NSFW Image Scanner")
        self.root.geometry("800x600")

        # Initialize components
        self.model_manager = None
        self.scanner = None
        self.scan_thread = None

        # Variables
        self.folder_path = tk.StringVar()
        self.threshold = tk.DoubleVar(value=0.5)
        self.max_workers = tk.IntVar(value=4)

        self.setup_gui()
        self.setup_models()

    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="NSFW Image Scanner", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Folder selection
        ttk.Label(main_frame, text="Folder to scan:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.folder_path, width=60).grid(row=1, column=1, 
                                                                          sticky=(tk.W, tk.E), pady=5, padx=(5, 5))
        ttk.Button(main_frame, text="Browse", command=self.browse_folder).grid(row=1, column=2, pady=5)

        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Detection Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        settings_frame.columnconfigure(1, weight=1)

        # Threshold setting
        ttk.Label(settings_frame, text="NSFW Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))

        threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=0.9, 
                                   variable=self.threshold, orient=tk.HORIZONTAL)
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        threshold_label = ttk.Label(threshold_frame, text="50%")
        threshold_label.pack(side=tk.RIGHT, padx=(10, 0))

        def update_threshold_label(*args):
            threshold_label.config(text=f"{self.threshold.get():.0%}")

        self.threshold.trace('w', update_threshold_label)

        # Workers setting
        ttk.Label(settings_frame, text="Parallel workers:").grid(row=1, column=0, sticky=tk.W, pady=2)
        workers_spinbox = ttk.Spinbox(settings_frame, from_=1, to=16, 
                                     textvariable=self.max_workers, width=10)
        workers_spinbox.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))

        # Scan button
        self.scan_button = ttk.Button(main_frame, text="Start Scan", 
                                     command=self.start_scan, style="Accent.TButton")
        self.scan_button.grid(row=3, column=0, columnspan=3, pady=20)

        # Progress bar
        self.progress_var = tk.StringVar(value="Ready to scan")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=4, column=0, columnspan=3, pady=5)

        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Scan Results", padding="10")
        results_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)

        # Results summary
        self.results_summary = ttk.Label(results_frame, text="No scan completed yet")
        self.results_summary.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Buttons frame
        buttons_frame = ttk.Frame(results_frame)
        buttons_frame.grid(row=2, column=0, pady=(10, 0))

        self.html_button = ttk.Button(buttons_frame, text="Generate HTML Report", 
                                     command=self.generate_html_report, state=tk.DISABLED)
        self.html_button.pack(side=tk.LEFT, padx=(0, 10))

        self.view_button = ttk.Button(buttons_frame, text="View Report in Browser", 
                                     command=self.view_html_report, state=tk.DISABLED)
        self.view_button.pack(side=tk.LEFT)

    def setup_models(self):
        """Initialize the model manager in a separate thread"""
        def setup_thread():
            try:
                self.progress_var.set("Setting up AI models...")
                self.progress_bar.start()

                self.model_manager = NSFWModelManager()
                self.model_manager.download_and_setup_models(
                    progress_callback=lambda msg: self.progress_var.set(msg)
                )
                self.scanner = ImageScanner(self.model_manager)

                self.progress_bar.stop()
                self.progress_var.set("Models loaded successfully. Ready to scan!")

            except Exception as e:
                self.progress_bar.stop()
                self.progress_var.set("Failed to setup models. Check console for details.")
                messagebox.showerror("Setup Error", f"Failed to setup AI models:\n{e}")
                logger.error(f"Model setup failed: {e}")

        threading.Thread(target=setup_thread, daemon=True).start()

    def browse_folder(self):
        """Open folder selection dialog"""
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)

    def start_scan(self):
        """Start the scanning process"""
        if not self.folder_path.get():
            messagebox.showerror("Error", "Please select a folder to scan")
            return

        if not self.model_manager:
            messagebox.showerror("Error", "AI models are not ready yet. Please wait.")
            return

        if not os.path.exists(self.folder_path.get()):
            messagebox.showerror("Error", "Selected folder does not exist")
            return

        # Disable scan button during scan
        self.scan_button.config(state=tk.DISABLED)
        self.progress_bar.start()

        # Start scan in separate thread
        self.scan_thread = threading.Thread(target=self.run_scan, daemon=True)
        self.scan_thread.start()

    def run_scan(self):
        """Run the actual scan (called in separate thread)"""
        try:
            def progress_callback(message):
                self.root.after(0, lambda: self.progress_var.set(message))

            results = self.scanner.scan_folder(
                self.folder_path.get(),
                threshold=self.threshold.get(),
                progress_callback=progress_callback,
                max_workers=self.max_workers.get()
            )

            # Update GUI in main thread
            self.root.after(0, lambda: self.scan_completed(results))

        except Exception as e:
            error_msg = f"Scan failed: {e}"
            logger.error(error_msg)
            self.root.after(0, lambda: self.scan_failed(error_msg))

    def scan_completed(self, results):
        """Handle scan completion (called in main thread)"""
        self.progress_bar.stop()
        self.scan_button.config(state=tk.NORMAL)

        # Update results summary
        summary_text = f"Scan completed: {len(results)} NSFW images found"
        self.results_summary.config(text=summary_text)
        self.progress_var.set("Scan completed successfully!")

        # Update results display
        self.results_text.delete(1.0, tk.END)

        if not results:
            self.results_text.insert(tk.END, "No NSFW images detected above the threshold.\n\n")
            self.results_text.insert(tk.END, "This could mean:\n")
            self.results_text.insert(tk.END, "• No inappropriate content was found\n")
            self.results_text.insert(tk.END, "• The threshold is too high\n")
            self.results_text.insert(tk.END, "• The images don't contain detectable NSFW content\n")
        else:
            self.results_text.insert(tk.END, f"Found {len(results)} flagged images:\n\n")

            for i, result in enumerate(results, 1):
                confidence_pct = result.overall_nsfw_probability * 100
                file_size = self.format_file_size(result.file_size_bytes)

                self.results_text.insert(tk.END, f"{i}. {result.filename}\n")
                self.results_text.insert(tk.END, f"   Confidence: {confidence_pct:.1f}%\n")
                self.results_text.insert(tk.END, f"   Size: {file_size}, Dimensions: {result.image_width}×{result.image_height}\n")
                self.results_text.insert(tk.END, f"   Model: {result.model_used}\n")
                self.results_text.insert(tk.END, f"   Path: {result.file_path}\n\n")

        # Enable report buttons
        if results:
            self.html_button.config(state=tk.NORMAL)
            self.view_button.config(state=tk.NORMAL)

    def scan_failed(self, error_message):
        """Handle scan failure (called in main thread)"""
        self.progress_bar.stop()
        self.scan_button.config(state=tk.NORMAL)
        self.progress_var.set("Scan failed!")
        messagebox.showerror("Scan Error", error_message)

    def generate_html_report(self):
        """Generate and save HTML report"""
        if not self.scanner or not self.scanner.results:
            messagebox.showerror("Error", "No scan results available")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
            title="Save HTML Report"
        )

        if save_path:
            try:
                self.scanner.generate_html_report(save_path)
                self.html_report_path = save_path
                messagebox.showinfo("Success", f"HTML report saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate HTML report:\n{e}")

    def view_html_report(self):
        """Open the HTML report in default browser"""
        if hasattr(self, 'html_report_path') and os.path.exists(self.html_report_path):
            webbrowser.open(f"file://{os.path.abspath(self.html_report_path)}")
        else:
            messagebox.showerror("Error", "Please generate the HTML report first")

    def format_file_size(self, size_bytes):
        """Format file size in human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def check_dependencies():
    """
    Check and install required dependencies
    """
    required_packages = [
        "torch",
        "torchvision", 
        "transformers",
        "opennsfw2",
        "timm",
        "pillow",
        "opencv-python",
        "numpy"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == "pillow":
                __import__("PIL")
            elif package == "opencv-python":
                __import__("cv2")
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def main():
    """Main application entry point"""
    print("NSFW Image Scanner v1.0")
    print("========================")
    print()

    # Check dependencies
    if not check_dependencies():
        print("\nPlease install the required dependencies and try again.")
        return

    # Check for GPU support
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"GPU acceleration available: {torch.cuda.get_device_name()}")
    else:
        print("Running on CPU (GPU acceleration not available)")

    print()

    # Start GUI
    try:
        app = NSFWScannerGUI()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
