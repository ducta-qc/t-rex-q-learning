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
from t_rex_tf import TRexGaming
import threading
import scipy.misc


KEYCODES = {
    'JUMP': ['38', '32'],
    'DUCK': ['40'],
    'RESTART': ['13']
}

FRAME_QUEUE = []
TRAIN_DATA_QUEUE = Queue.Queue()
GAME_OVER = 0
GAME_PLAYING = 1
FRAME_CHANNELS = 4
T_REX = None

class BotNamespace(BaseNamespace, BroadcastMixin):
    def __init__(self, *args, **kwargs):
        super(BotNamespace, self).__init__(*args, **kwargs)
        self.frame_queue = []
        self.last_frame_tuple = None
        self.prev_state = None
        self.curr_state = None
        self.is_learning = False

    def decode_action(self, action):
        if action == 'NONE': 
            return 0
        elif action == 'JUMP': 
            return 1
        else: #action == 'DUCK'
            return 2

    def encode_action(self, action):
        if action == 0:
            return 0
        elif action == 1:
            return '38' # Javascript keyCode for the up arrow key
        else:
            return '40' # Javascript keyCode for the down arrow key

    def combine_frames(self, frames):
        c0 = frames[0][0]
        c1 = frames[1][0]
        c2 = frames[2][0]
        c3 = frames[3][0]
        test = np.stack([c0, c1, c2, c3], axis=2)
        return test

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
        resized_frame = cv2.resize(gray_frame, (80, 80))

        action = self.decode_action(action)

        #cv2.imshow('frame', gray_frame)

        reward = 0.
        if game_status == GAME_OVER:
            reward = -100.
            print('reward', reward)
        self.frame_queue.append((resized_frame, action, reward))

        if len(self.frame_queue) == 4:
            self.prev_state = self.curr_state
            self.curr_state = self.combine_frames(self.frame_queue)

            if not self.prev_state is None:
                TRAIN_DATA_QUEUE.put([self.prev_state, action, reward, self.curr_state])

            self.frame_queue.pop(0)

        if not self.curr_state is None:
            if game_status == GAME_PLAYING:
                ext_curr_state = self.curr_state[np.newaxis,:,:,:]
                next_action = int(T_REX.take_a_action(ext_curr_state))
                print('next_action', next_action)
                next_action = self.encode_action(next_action)
            else:
                next_action = "13"
                self.prev_state = None
                self.curr_state = None
                self.frame_queue = []
            self.emit('recv_action', {"action": next_action})

    def recv_connect(self):
        pass


def socketio_service(request):
    socketio_manage(request.environ,
                    {'/bot': BotNamespace},
                    request=request)

    return Response('')

if __name__ == '__main__':
    try:
        T_REX = TRexGaming(TRAIN_DATA_QUEUE, using_cuda=False)
        T_REX.learning()
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
        #server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
