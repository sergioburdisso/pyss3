# -*- coding: utf-8 -*-
"""Tests for pyss3.server."""
import pyss3.server as s
import threading
import argparse
import socket
import pytest
import pyss3
import json
import sys

from os import path
from pyss3 import SS3
from pyss3.util import Dataset, Print

HTTP_REQUEST = "%s %s HTTP/1.1\r\nContent-Length: %d\r\n\r\n%s"
RECV_BUFFER = 1024 * 1024  # 1MB

PYTHON3 = sys.version_info[0] >= 3
DATASET_FOLDER = "dataset"
DATASET_FOLDER_MR = "dataset_mr"
ADDRESS, PORT = "localhost", None
LT = s.Live_Test

dataset_path = path.join(path.abspath(path.dirname(__file__)), DATASET_FOLDER)
dataset_path_mr = path.join(path.abspath(path.dirname(__file__)), DATASET_FOLDER_MR)

x_train, y_train = None, None
clf = None

pyss3.set_verbosity(0)

x_train, y_train = Dataset.load_from_files(dataset_path_mr)
x_train, y_train = Dataset.load_from_files(dataset_path, folder_label=False)
clf = SS3()

clf.fit(x_train, y_train)

LT.serve()  # no model error
LT.set_model(clf)
LT.get_port()


class MockCmdLineArgs:
    """Mocked command-line arguments."""

    quiet = True
    MODEL = "name"
    path = dataset_path
    label = 'folder'
    port = 0


@pytest.fixture()
def mockers(mocker):
    """Set mockers up."""
    mocker.patch.object(LT, "serve")
    mocker.patch.object(SS3, "load_model")
    mocker.patch.object(argparse.ArgumentParser, "add_argument")
    mocker.patch.object(argparse.ArgumentParser,
                        "parse_args").return_value = MockCmdLineArgs


@pytest.fixture(params=[0, 1, 2, 3])
def test_case(request, mocker):
    """Argument values generator for test_live_test(test_case)."""
    mocker.patch("webbrowser.open")

    if request.param == 0:
        LT.set_testset_from_files(dataset_path, folder_label=False)
    elif request.param == 1:
        LT.set_testset_from_files(dataset_path_mr, folder_label=True)
    elif request.param == 2:
        LT.set_testset(x_train, y_train)
    else:
        LT.__server_socket__ = None

    yield request.param


def http_request(path, body='', get=False, as_bytes=False):
    """Create a basic HTTP request message."""
    request = HTTP_REQUEST % ("GET" if get else "POST", path, len(body), body)
    return request.encode() if as_bytes else request


def http_response_body(sock):
    """Return all HTTP message body."""
    data = sock.recv(RECV_BUFFER).decode()
    length = s.get_http_contlength(data)
    body = s.get_http_body(data)
    while len(body) < length and data:
        data = sock.recv(RECV_BUFFER).decode()
        body += data
    return body  # url_decode(body)


def send_http_request(path, body='', get=False, json_rsp=True):
    """Send an HTTP  request to the Live Test Server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ADDRESS, PORT))
    sock.sendall(http_request(path, body, get, as_bytes=True))
    r = http_response_body(sock)
    sock.close()
    return json.loads(r) if json_rsp and r else r


def test_http_helper_functions():
    """Test for pyss3.server HTTP helper function."""
    assert s.content_type("js") == "application/javascript"
    assert s.content_type("non-existing") == "application/octet-stream"

    request_path = "/the/path"
    request_body = "the body"
    assert s.parse_and_sanitize("../../a/path/../../")[0][-17:] == "a/path/index.html"
    assert s.parse_and_sanitize("/")[0][-10:] == "index.html"
    assert s.get_http_path(http_request(request_path)) == request_path
    assert s.get_http_body(http_request("", request_body)) == request_body
    assert s.get_http_contlength(http_request("", request_body)) == len(request_body)


def test_live_test(test_case):
    """Test the HTTP Live Test Server."""
    global PORT

    if test_case != 3:
        PORT = LT.start_listening()
    else:
        Print.error = lambda _: None  # do nothing

    serve_args = {
        "x_test": x_train if test_case == 2 or test_case == 3 else None,
        "y_test": y_train if test_case == 2 else None,
        "quiet": test_case != 0
    }

    if PYTHON3:
        threading.Thread(target=LT.serve, kwargs=serve_args, daemon=True).start()
    else:
        return
        # threading.Thread(target=LT.serve, kwargs=serve_args).start()

    if test_case == 3:
        return

    # empty message
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ADDRESS, PORT))
    sock.sendall(b'')
    sock.close()

    # decode error
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ADDRESS, PORT))
    sock.sendall(b'\x01\x0E\xFF\xF0\x02\x0F\xE1')
    sock.close()

    # 404 error
    send_http_request("/404")

    # ack
    send_http_request("/ack")

    # get_info
    r = send_http_request("/get_info")
    assert r["model_name"] == clf.get_name()
    cats = r["categories"]
    docs = r["docs"]
    assert len(cats) == 8 + 1
    # assert len(docs) == len(cats) - 1
    # assert len(docs[cats[0]]["path"]) == 100

    # classify
    r = send_http_request(
        "/classify",
        "this is an android mobile " * (1024 * 4 if test_case == 0 else 1)
    )
    assert r["ci"][r["cvns"][0][0]] == "science&technology"

    # get_doc
    for c in docs:
        r = send_http_request("/get_doc", docs[c]["path"][1])
        assert len(r["content"][:2]) == 2

    # GET 404
    send_http_request("/404", get=True, json_rsp=False)

    # GET index.html
    r = send_http_request("/", get=True, json_rsp=False)
    assert "<html>" in r


def test_main(mockers):
    """Test the main() function."""
    if not PYTHON3:
        return

    s.main()
