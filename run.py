import time
import os
import subprocess
import os
import subprocess
import sys
import datetime
import signal
import shutil
import cv2
#from modelscope.hub.snapshot_download import snapshot_download

def read_config():
	
    print("read_config")

    myvars = {}
    with open("/home/meldib/work/projects/DDColor/config.txt") as myfile:
        for line in myfile:
            key, val = line.partition("=")[::2]
            myvars[key.strip()] = val

    os.environ["PIP_CONF"] = myvars["PIP_CONF"].strip()
    os.environ["NOISE_LEVEL"] = myvars["NOISE_LEVEL"].strip()
    # drunet_color | drunet_gray
    os.environ["DENOISE_MODEL"] = myvars["DENOISE_MODEL"].strip()
    # offline | online
    os.environ["ViDEO_SOURCE"] = myvars["ViDEO_SOURCE"].strip()
    # offline | online
    os.environ["AUDIO_SOURCE"] = myvars["AUDIO_SOURCE"].strip()
    os.environ["RENDER_FACTOR"] = myvars["RENDER_FACTOR"].strip()
    os.environ["SPLIT_DURATION"] = myvars["SPLIT_DURATION"].strip()
    os.environ["VIDEO_TO_PROCESS"] = myvars["VIDEO_TO_PROCESS"].strip()
    os.environ["TO_SPLIT"] = myvars["TO_SPLIT"].strip()
    os.environ["VIDEO_LINK"] = myvars["VIDEO_LINK"].strip()
    os.environ["HAS_AUDIO"] = myvars["HAS_AUDIO"].strip()
    os.environ["AUDIO_LINK"] = myvars["AUDIO_LINK"].strip()
    os.environ["DRIVE_PATH"] = myvars["DRIVE_PATH"].strip()
    os.environ["PROCESS_STAGE"] = "clearize"
    os.environ["IMG_INPUT1"] = ""
    os.environ["IMG_INPUT2"] = ""

    if os.environ.get("DENOISE_MODEL").strip() == "drunet_color":
        os.environ["DENOISE_RESULT"] = "set_real_dn_drunet_color"
    else:
        os.environ["DENOISE_RESULT"] = "set_real_dn_drunet_gray"

def run_command(command):

    subprocess.call(command, shell=True)

def face_enhance():

	os.environ["PROCESS_STAGE"] = "face_enhance"
	run_command("mv /home/meldib/work/projects/DDColor/test_imgs  /home/meldib/work/projects/DDColor/GPEN-colab/examples/imgs")
	#run_command("cd /home/meldib/work/projects/DDColor/GPEN-colab && . ~/.zshrc && . activate gpen && python demo.py --task FaceEnhancement --model GPEN-BFR-512 --in_size 512 --channel_multiplier 2 --narrow 1 --use_sr --use_cuda --indir /home/meldib/work/projects/DDColor/GPEN-colab/examples/imgs --outdir /home/meldib/work/projects/DDColor/GPEN-colab/results/outs-BFR")
	run_command("cd /home/meldib/work/projects/DDColor/GPEN-colab  && . ~/.zshrc && . activate gpen  && python demo.py --task FaceEnhancement --model GPEN-BFR-256 --in_size 256 --channel_multiplier 1 --narrow 0.5 --sr_scale 1 --use_sr --use_cuda --indir /home/meldib/work/projects/DDColor/GPEN-colab/examples/imgs --outdir /home/meldib/work/projects/DDColor/GPEN-colab/results/outs-BFR")

	run_command("mv /home/meldib/work/projects/DDColor/GPEN-colab/results/outs-BFR  /home/meldib/work/projects/DDColor/input_images1")

	#run_command("rm -rf /home/meldib/work/projects/DDColor/GPEN-colab/examples/imgs")



def process():
    os.environ["PROCESS_STAGE"] = "clearize"
    read_config()
    face_enhance()


  
process()




