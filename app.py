from flask import Flask,abort,request
from flask import Flask, request, jsonify,g
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import json
import logging
import datetime
import dlib
import os
import cv2
import pyramids
import heartrate
import preprocessing
import eulerian
import error_codes

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)
handler = logging.FileHandler('hr_api.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(lineno)d-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)
log.info("The Medtek HR is Started")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['mp4'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
def return_response_new(data, message, error:bool, error_code=None):
    if error:
        _response_json = {
            'data': data,
            'message':message,
            'status': not error,
            'error_code': error_code
        }
    else:
        _response_json = {
            'data':data,
            'message':message,
            'status': not error,
            'status_code': error_code
        }
    log.info("Final Response from HR model: {}".format(_response_json))
    return jsonify(_response_json)

##################### HR Calculation in which we pass video###############
def PredictHR(video_path):
    try:
        # Frequency range for Fast-Fourier Transform
        freq_min = 1
        freq_max = 1.8

        # Preprocessing phase
        print("Reading + preprocessing video...")
        video_frames, frame_ct, fps = preprocessing.read_video(video_path)


        # Build Laplacian video pyramid
        print("Building Laplacian video pyramid...")
        lap_video = pyramids.build_video_pyramid(video_frames)
        

        amplified_video_pyramid = []

        for i, video in enumerate(lap_video):
            if i == 0 or i == len(lap_video)-1:
                continue

            # Eulerian magnification with temporal FFT filtering
            print("Running FFT and Eulerian magnification...")
            result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
    
            lap_video[i] += result

            # Calculate heart rate
            print("Calculating heart rate...")
            heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)


        # Collapse laplacian pyramid to generate final video
        print("Rebuilding final video...")
        amplified_frames = pyramids.collapse_laplacian_video_pyramid(lap_video, frame_ct)
        # amplified_frames = collapse_laplacian_video_pyramid(lap_video, frame_ct)

        # Output heart rate and final video
        print("Heart rate: ", heart_rate, "bpm")
        log.info("HR: {}".format(heart_rate))
        return True, {"HR":round(heart_rate)}
    except Exception as e:
        return False,e
    
def request_file(request):
    handle = request.files.get('video_file')
    log.info(handle)
    file = request.files['video_file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    
    return os.path.join(app.config['UPLOAD_FOLDER'],filename)

@app.route("/v1/predict_hr",methods = ["POST"])
def predict_hr():
    try:
        if 'txt_id' not in request.form:
            return return_response_new(data=None,message='Transaction ID not found.',
                                      error_code=error_codes.INVALID_SESSION_ID, error=True)
        session_id = request.form.get('txt_id').strip().lower()
        
        if session_id is None:
            return return_response_new(data=None, message='Please pass valid session_id',
                                      error=True,error_code=error_codes.INVALID_SESSION_ID)
        log.info(session_id + "<<<<< Inside Predict HR API POST>>>>>>>>")
        
        if request.method == 'POST':
            '''##### Video File Check #######'''
            log.info("Inside POST")
            
            if not request.files:
                return return_response_new(data=None, message='Key Live video file not found',
                                          error_code=error_codes.LIVE_VIDEO_FILE_NOT_FOUND,error=True)
            log.info(request.files)
            live_video = request.files.get('video_file')
            
            log.info(live_video)
            if live_video:
                live_video_full_path = request_file(request)
                log.info(session_id + 'live video save to path - {}'.format(live_video_full_path))
                status, output = PredictHR(live_video_full_path)
                if status:
                    return return_response_new(data={"hr_values":output},message='Heart Rate values Calculated Sucessfully',
                                              error=False, error_code="200")
                else:
                    log.debug(output)
                    return return_response_new(data=None,message="Error in calculating HR value", error=True,
                                              error_code=error_codes.UNKNOWN_ERROR_OCCURRED)
    except Exception as e:
        log.debug(e)
        return return_response_new(data=None,message="Oops Something Went Wrong", error=True,
                                  error_code=error_codes.UNKNOWN_ERROR_OCCURRED)
    
if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)