# NSFW Image Scanner Setup Guide
========================================

This guide will help you install and run the NSFW Image Scanner application.

## Prerequisites

- Python 3.8 or higher
- At least 4GB of RAM (8GB+ recommended)
- Optional: NVIDIA GPU with CUDA support for faster processing

## Installation Steps

### 1. Clone or Download the Files

Ensure you have these files in a folder:
- `nsfw_image_scanner.py` (main application)
- `requirements.txt` (dependencies)
- `README.md` (this file)

### 2. Install Python Dependencies

Open a terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

For GPU acceleration (if you have an NVIDIA GPU with CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Run the Application

```bash
python nsfw_image_scanner.py
```

## First Run Setup

When you first run the application:

1. **Model Download**: The app will automatically download AI models (~500MB-1GB total)
2. **GPU Detection**: It will detect if GPU acceleration is available
3. **Ready to Use**: Once models are loaded, you can start scanning

## How to Use

### Basic Scanning Process

1. **Select Folder**: Click "Browse" to choose a folder containing images
2. **Adjust Settings**:
   - **NSFW Threshold**: Lower = more sensitive (catches more), Higher = less sensitive
   - **Parallel Workers**: More workers = faster processing (but uses more resources)
3. **Start Scan**: Click "Start Scan" to begin processing
4. **View Results**: Flagged images will appear in the results panel
5. **Generate Report**: Click "Generate HTML Report" for a detailed visual report

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- GIF (.gif)
- WebP (.webp)
- TIFF (.tiff, .tif)

## Understanding Results

### Confidence Scores
- **80-100%**: High confidence (very likely NSFW)
- **60-79%**: Medium confidence (probably NSFW)
- **50-59%**: Low confidence (possibly NSFW)

### Categories Detected
The scanner detects general NSFW content. Specific body part detection depends on the model used:
- **General NSFW**: Overall inappropriate content
- **Explicit Content**: Sexual or nude content

## Performance Tips

### For Large Image Collections (1000+ images)

1. **Use GPU**: Enable GPU acceleration for 5-10x speed improvement
2. **Adjust Workers**: Set parallel workers to match your CPU cores
3. **Batch Processing**: Process folders in smaller batches if memory is limited
4. **SSD Storage**: Use SSD for faster image loading

### Memory Usage

- **CPU Mode**: ~2-4GB RAM
- **GPU Mode**: ~2-6GB GPU VRAM (depending on model)
- **Large Batches**: May require more memory

## Troubleshooting

### Common Issues

#### "No module named 'torch'"
```bash
pip install torch torchvision
```

#### "CUDA out of memory"
- Reduce parallel workers
- Use CPU mode instead
- Close other GPU-intensive applications

#### "Failed to setup AI models"
- Check internet connection (models download on first run)
- Ensure sufficient disk space (~2GB free)
- Try running as administrator (Windows) or with sudo (Linux)

#### GUI doesn't appear
- Ensure tkinter is installed: `pip install tk`
- On Linux: `sudo apt-get install python3-tk`

### Model Download Issues

If models fail to download:
1. Check firewall/antivirus settings
2. Try manual download from Hugging Face
3. Use VPN if region-blocked

## File Structure

After running, the app creates:
```
your_scan_folder/
├── .nsfw_thumbnails/          # Thumbnail cache
│   ├── thumb_image1.jpg
│   └── thumb_image2.jpg
└── your_images...

~/.nsfw_scanner/               # User data folder
└── models/                    # Downloaded AI models
    ├── falconsai_model/
    └── other_models/
```

## Privacy & Security

### Data Privacy
- **100% Local Processing**: No images or data sent to cloud services
- **No Network Access**: After model download, works offline
- **No Data Collection**: Application doesn't collect or transmit user data

### Security Features
- Models downloaded from official repositories only
- No external API calls during scanning
- Thumbnails stored locally and can be deleted

## Advanced Usage

### Command Line Options

For advanced users, you can modify the script to accept command-line arguments:

```python
# Example: Scan folder from command line
python nsfw_image_scanner.py --folder "/path/to/images" --threshold 0.7
```

### Custom Models

To use different NSFW detection models, modify the `NSFWModelManager` class:

```python
# Add custom model in NSFWModelManager.download_and_setup_models()
# Follow the existing pattern for new models
```

## Technical Details

### AI Models Used

1. **Falconsai/nsfw_image_detection** (Primary)
   - Vision Transformer (ViT) based
   - 96%+ accuracy
   - Best general performance

2. **Marqo/nsfw-image-detection-384** (Backup)
   - Lightweight and fast
   - 98.5% accuracy on test set
   - 20x smaller than alternatives

3. **OpenNSFW2** (Fallback)
   - Classic Yahoo open-source model
   - TensorFlow 2 implementation
   - Reliable baseline

### Performance Benchmarks

On typical hardware:
- **RTX 4080**: ~50-100 images/second
- **CPU (Intel i7)**: ~5-15 images/second
- **Memory Usage**: 2-6GB depending on batch size

## Support

### Getting Help

1. Check this README for common solutions
2. Review the console output for error messages
3. Ensure all dependencies are correctly installed
4. Try running with different settings (lower threshold, fewer workers)

### Reporting Issues

When reporting problems, include:
- Operating system and Python version
- Complete error message
- Hardware specifications (CPU/GPU)
- Steps to reproduce the issue

## License

This project is released under the MIT License. The AI models used may have their own licenses - please check their respective repositories.

## Disclaimer

This tool is intended for legitimate content moderation and personal use only. Users are responsible for complying with applicable laws and regulations in their jurisdiction. The accuracy of AI models is not guaranteed, and manual review is recommended for critical applications.
