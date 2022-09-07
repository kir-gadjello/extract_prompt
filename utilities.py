#!/usr/bin/env python3
# Prompt Extractor v0.4
# Copyright (c) 2022 kir-gadjello

import os
import requests
from tqdm import tqdm
import hashlib
import base64
from torch import device


def touch_model(model, args):
    if args.device != "cpu":
        model = model.to(device(args.device))
    if args.device != "cpu" and args.half:
        model = model.half()
    model.eval()
    return model


def chunk(seq, size):
    ret = []
    t = []
    for item in seq:
        t.append(item)
        if len(t) >= size:
            ret.append(t)
            t = []
    if len(t) > 0:
        ret.append(t)
    return ret


def make_weightlist_parser(init_weights, num=int):
    def parse_weightlist(s):
        if len(s) == 0:
            return dict(**init_weights)

        if s.isdigit():
            ret = dict(**init_weights)
            for k in ret.keys():
                ret[k] = num(s)
            return ret

        def parse_w(ss):
            a, b = ss.split(":")
            return a, num(b)

        ret = dict(**init_weights)

        for k, v in list(map(parse_w, s.split(","))):
            ret[k] = v

        return ret

    return parse_weightlist


def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 4096
    r = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        pbar = tqdm(unit="B", total=int(r.headers["Content-Length"]))
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk:
                # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)
    return filename


def ensure_file(url, filename=None, dstdir="./", size=None):
    assert filename is not None
    fpath = os.path.join(dstdir, filename)
    if os.path.isfile(fpath):
        if size is None:
            return

        fsize = os.stat(fpath).st_size
        if fsize == size:
            return
        elif fsize != size:
            print(f"file size mismatch! {fsize} vs expected {size}")

    print("DL:", url, "->", fpath)

    download_file(url, fpath)


def load_list(filename):
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        items = [line.strip() for line in f.readlines()]
    return items


def hash_arr(arr):
    hashGen = hashlib.sha512()
    for elem in arr:
        hashGen.update(elem.encode("utf-8"))
    return base64.urlsafe_b64encode(hashGen.digest()).decode()


printed = {}


def print_once(s, label, printer=print):
    global printed
    if not printed.get(label):
        printer(s)
        printed[label] = True
