from flask import render_template, Flask, request, send_from_directory, flash, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
# set the project root directory as the static folder, you can set others.

app = Flask(__name__, static_url_path='')
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

UPLOAD_FOLDER = 'assets/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/')
def index():
    return render_template("index.html", error='', success=False)

@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory("assets", path)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'files' not in request.files:
            print('No file part')
            return render_template("index.html", error="No upload file found!")
        file = request.files['files']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return render_template("index.html", error="No upload file found!")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_trans = removeImageBackground(filename)
            return render_template("index.html", beforeimage=filename, image=file_trans, success=True)
    
    return render_template("index.html", error="No upload file found!")

def removeImageBackground(image) :
    # load image
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], image))
    # convert to graky
    hh, ww = img.shape[:2]
    # threshold on white
    # Define lower and uppper limits
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])
    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # invert morp image
    mask = 255 - morph
    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)
    # save resulting masked image
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'trans_'+image), result)
    # display result, though it won't show transparency
    # cv2.imshow("INPUT", img)
    # cv2.imshow("GRAY", gray)
    # cv2.imshow("MASK", mask)
    # cv2.imshow("RESULT", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return 'trans_'+image
    
def removeImageBackgroud2(image) :
    img= os.path.join(app.config['UPLOAD_FOLDER'], image);
    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'trans2_'+image), dst)
    return 'trans2_'+image
    
if __name__ == "__main__":
    app.debug = True
    app.run()