[![Build Status](https://travis-ci.com/DACUS1995/License-Plate-Recognition.svg?token=EXzDMzSxfmwgYPg9Ttfx&branch=main)](https://travis-ci.com/DACUS1995/License-Plate-Recognition)

# License-Plate-Recognition

This license plate recognition projects uses FastAPI framework to server the actual detection functionality. Behind the scenes it uses the tesseract OCR engine to detect the characters in a preprocessed and localized license plate. In the future I will add more OCR methods.

---

Instalation:
(for Windows https://tesseract-ocr.github.io/tessdoc/Installation.html#windows):
```bash
sudo apt-get install tesseract-ocr (Ubuntu)
pip install -r requirements.txt
```

To run the server:
```bash
./run_dev.cmd
```

or

```bash
uvicorn main:app
```

To run the test:
```python
python test.py
```