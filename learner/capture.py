#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from wsgiref.simple_server import make_server
from socketio.server import SocketIOServer
from pyramid.paster import get_app
from gevent import monkey; monkey.patch_all()
from pyramid.config import Configurator
from pyramid.response import Response
import cv2
import numpy as np
from socketio import socketio_manage
from socketio.namespace import BaseNamespace
from socketio.mixins import BroadcastMixin
import Queue

KEYCODES = {
    'JUMP': ['38', '32'],
    'DUCK': ['40'],
    'RESTART': ['13']
}

FRAME_QUEUE = []
TRAIN_DATA_QUEUE = []
GAME_OVER = 0
GAME_PLAYING = 1
FRAME_CHANNELS = 4


class BotNamespace(BaseNamespace, BroadcastMixin):
    def __init__(self, *args, **kwargs):
        super(BotNamespace, self).__init__(*args, **kwargs)
        self.frame_queue = []
        self.train_data_queue = Queue.Queue()

    def on_recv_frame(self, data):
        frame = data['frame']
        action = data['action']
        game_status = data['game_status']

        # Decode frame and put it in queue
        frame = frame.replace('data:image/png;base64,', '')
        frame += '='
        frame = frame.decode("base64")
        frame = np.fromstring(frame, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if action == 'NONE': 
            action = 0
        elif action == 'JUMP': 
            action = 1
        else action == 'DUCK':
            action = 2

        reward = 0.
        if game_status == GAME_OVER:
            reward = -100.
        self.frame_queue.append((gray_frame, action, reward))
        #self.emit('recv_action', {"action": "38"})

        for i in range(len(self.frame_queue) - 1, len(self.frame_queue) - FRAME_CHANNELS, -1):
            self.train_data_queue.append([
                    self.frame_queue[i], self.frame_queue[i-1], 
                    self.frame_queue[i-2], self.frame_queue[i-3]
            ])

    def recv_connect(self):
        pass


def socketio_service(request):
    socketio_manage(request.environ,
                    {'/bot': BotNamespace},
                    request=request)

    return Response('')


# def recv_game_frame(request):
#     frame_data = request.POST['frame']
#     frame_data = frame_data.replace('data:image/png;base64,', '');
#     frame_data += '='
#     binary_png = frame_data.decode("base64")
#     imarr = np.fromstring(binary_png, np.uint8)
#     img_ = cv2.imdecode(imarr, cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     return {}

if __name__ == '__main__':
    try:
        config = Configurator()
        config.add_route('socket_io', 'socket.io/*remaining')
        config.add_view(socketio_service, route_name='socket_io')
        #config.add_route('recv_frame', 'recv/')
        #config.add_view(recv_game_frame, route_name='recv_frame', request_method='POST')
        config.add_static_view('/', 't_rex')
        app = config.make_wsgi_app()
        SocketIOServer(('0.0.0.0', 1234), app,
                       resource="socket.io", policy_server=True,
                       policy_listener=('0.0.0.0', 10843)).serve_forever()

        # create named window
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
