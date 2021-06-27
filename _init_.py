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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold input image as mask
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    # negate mask
    mask = 255 - mask
    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
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
    
if __name__ == "__main__":
    app.debug = True
    app.run()