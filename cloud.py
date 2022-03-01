import sys
import os
import argparse
import moxing as mox
import logging
import re
import fnmatch
import subprocess

logging.basicConfig(level=logging.INFO)
# copy program to mox
# mox.file.copy_parallel("s3://mt-codes/marian", "/cache/marian")
# os.system("chmod +x /cache/marian/marian*")

os.environ['DLS_LOCAL_CACHE_PATH'] = "/cache"
mox.file.make_dirs("/cache")

LOCAL_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
assert mox.file.exists(LOCAL_DIR)
logging.info("local disk: " + LOCAL_DIR)


parser = argparse.ArgumentParser()
parser.add_argument("--script", type=str, default="")
# parser.add_argument("--script_bin", type=str, default="")
# parser.add_argument("--script_train", type=str, default="")
# parser.add_argument("--script_average", type=str, default="")
# parser.add_argument("--script_generate", type=str, default="")
parser.add_argument("--src", type=str, default="")
parser.add_argument("--tgt", type=str, default="")
parser.add_argument("--max_update", type=str, default="")
parser.add_argument("--code_dir", type=str, default="s3://aarc-beijing4/code/litianhao/")
parser.add_argument("--data_url", type=str, default="")
parser.add_argument("--file_pattern", type=str, default=None)
parser.add_argument("--train_url", type=str, default="")
parser.add_argument("--dev_prefix", type=str, default="", required=True)
parser.add_argument("--train_prefix", type=str, default="", required=True)
parser.add_argument("--test_prefix", type=str,default="", required=True)
args, _ = parser.parse_known_args()
src_lang = args.src
tgt_lang = args.tgt
max_up = args.max_update
# copy to /cache
logging.info("copying " + args.code_dir)
mox.file.copy_parallel(
    args.code_dir, os.path.join(LOCAL_DIR,"code_dir"))

logging.info("copying " + args.train_url)
mox.file.copy_parallel(
    args.train_url, 
    os.path.join(LOCAL_DIR,"model_dir"))          
             
logging.info("copying data...")
train_set = []
dev_set = []
test_set = []
if args.data_url:
    logging.info("copying data...")
    local_data_dir = os.path.join(LOCAL_DIR, "data_dir")
    if args.train_prefix:
        file_list=[]
        prefix = args.train_prefix.split(",")+args.dev_prefix.split(",")+args.test_prefix.split(",")
        patterns = [p + "." + args.src for p in prefix] + [p + "." + args.tgt for p in prefix]
        print(patterns)
        for fname in mox.file.list_directory(args.data_url, recursive=True):
            for pattern in patterns:
                if fnmatch.fnmatch(fname, pattern):
                    file_list.append(fname)
                    break
        logging.info(file_list)
        mox.file.copy_parallel(
            args.data_url, 
            local_data_dir,
            file_list=file_list)
    else:
        logging.info(mox.file.list_directory(args.data_url, recursive=True))
        mox.file.copy_parallel(
            args.data_url, local_data_dir)
else:
    local_data_dir = args.data_url

# root_dir = os.path.dirname(os.path.abspath(__file__))
root_dir=os.path.join(LOCAL_DIR, "code_dir")+"/scripts"
script = os.path.join(root_dir, args.script)


logging.info("excuting ...")
cmd =["bash", script, src_lang, tgt_lang, max_up, args.train_prefix, args.dev_prefix, args.test_prefix]
logging.info(" ".join(cmd))
os.system(" ".join(cmd))

# script = os.path.join(root_dir, args.script_bin)

# logging.info("excuting ...")
# cmd =["bash", script, src_lang, tgt_lang]
# process = subprocess.Popen(" ".join(cmd), shell=True, stdout=sys.stdout)
# process.wait()

# script = os.path.join(root_dir, args.script_train)

# logging.info("excuting ...")
# cmd =["bash", script, src_lang, tgt_lang]
# process = subprocess.Popen(" ".join(cmd), shell=True, stdout=sys.stdout)
# process.wait()

# script = os.path.join(root_dir, args.script_average)

# logging.info("excuting ...")
# cmd =["bash", script, src_lang, tgt_lang]
# process = subprocess.Popen(" ".join(cmd), shell=True, stdout=sys.stdout)
# process.wait()

# script = os.path.join(root_dir, args.script_generate)

# logging.info("excuting ...")
# cmd =["bash", script, src_lang, tgt_lang]
# process = subprocess.Popen(" ".join(cmd), shell=True, stdout=sys.stdout)
# process.wait()

# copy back to s3
model_dir = args.train_url
data_dir = args.data_url
logging.info("copying output back...")
if not mox.file.exists(model_dir):
#     mox.file.remove(args.model_dir, recursive=True)
    mox.file.make_dirs(model_dir)

mox.file.copy_parallel(
    os.path.join(LOCAL_DIR, "model_dir"), 
    model_dir)

logging.info("end")