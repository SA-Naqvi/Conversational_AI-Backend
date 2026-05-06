import os
import urllib.request
import tarfile
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Base directory for backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "sherpa-onnx": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        "dest_dir": os.path.join(BASE_DIR, "sherpa-onnx-streaming-zipformer-en-2023-06-26"),
        "is_archive": True
    },
    "piper-amy-low": {
        "files": [
            {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx",
                "name": "en_US-amy-low.onnx"
            },
            {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx.json",
                "name": "en_US-amy-low.onnx.json"
            }
        ],
        "dest_dir": os.path.join(BASE_DIR, "piper-amy-low"),
        "is_archive": False
    }
}

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logger.info(f"File already exists: {dest_path}")
        return
    
    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        logger.info(f"Successfully downloaded {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")

def main():
    # 1. Download Sherpa-ONNX and Piper models
    for name, config in MODELS.items():
        os.makedirs(config["dest_dir"], exist_ok=True)
        
        if config.get("is_archive"):
            archive_name = os.path.basename(config["url"])
            archive_path = os.path.join(BASE_DIR, archive_name)
            download_file(config["url"], archive_path)
            
            if os.path.exists(archive_path):
                logger.info(f"Extracting {archive_name}...")
                try:
                    with tarfile.open(archive_path, "r:bz2") as tar:
                        # Extract everything into a temporary dir first to handle nested folders
                        temp_extract_dir = os.path.join(BASE_DIR, "temp_extract")
                        tar.extractall(path=temp_extract_dir)
                        
                        # Move the contents of the extracted folder to the dest_dir
                        extracted_folder = os.path.join(temp_extract_dir, name + "-streaming-zipformer-en-2023-06-26")
                        if os.path.exists(extracted_folder):
                            for item in os.listdir(extracted_folder):
                                s = os.path.join(extracted_folder, item)
                                d = os.path.join(config["dest_dir"], item)
                                if os.path.exists(d):
                                    if os.path.isdir(d): shutil.rmtree(d)
                                    else: os.remove(d)
                                shutil.move(s, d)
                            shutil.rmtree(temp_extract_dir)
                    os.remove(archive_path)
                    logger.info(f"Successfully extracted {name} and cleaned up.")
                except Exception as e:
                    logger.error(f"Extraction failed for {name}: {e}")
        else:
            for file_info in config["files"]:
                dest_path = os.path.join(config["dest_dir"], file_info["name"])
                download_file(file_info["url"], dest_path)

    # 2. Trigger Whisper and SentenceTransformer downloads (cached by library)
    logger.info("Triggering Whisper and SentenceTransformer model downloads (caching)...")
    try:
        from faster_whisper import WhisperModel
        WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("Whisper 'base' model cached.")
    except Exception as e:
        logger.warning(f"Could not cache Whisper model: {e}")

    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer 'all-MiniLM-L6-v2' model cached.")
    except Exception as e:
        logger.warning(f"Could not cache SentenceTransformer model: {e}")

if __name__ == "__main__":
    main()
