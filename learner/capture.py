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
import random

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
T_REX = None
GAME_TURN = 0
REPLAY_SIZE = 300
NUM_FRAME = 0

class BotNamespace(BaseNamespace, BroadcastMixin):
    def __init__(self, *args, **kwargs):
        super(BotNamespace, self).__init__(*args, **kwargs)
        self.frame_queue = []
        self.raw_frame_queue = []
        self.last_frame_tuple = None
        self.prev_state = None
        self.curr_state = None
        self.debug = True
        self.emit_action = True
        self.dispatch_action = 'NONE'

    def decode_action(self, action):
        if action == 'NONE': 
            return 0
        if action == 'JUMP': 
            return 1
        # else: #action == 'DUCK'
        #     return 2
    def num_to_action(self, action):
        if action == 0:
            return 'NONE'
        if action == 1:
            return 'JUMP'
        # if action == 2:
        #   return 'DUCK'

    def encode_action(self, action):
        if action == 0:
            return 0
        if action == 1:
            return '38' # Javascript keyCode for the up arrow key
        # else:
        #     return '40' # Javascript keyCode for the down arrow key

    def combine_frames(self, frames):
        c0 = frames[0]
        c1 = frames[1]
        c2 = frames[2]
        c3 = frames[3]
        s = np.stack([c0, c1, c2, c3], axis=2)
        return s

    def on_recv_frame(self, data):
        global GAME_TURN, NUM_FRAME
        frame = data['frame']
        #action = data['action']
        game_status = data['game_status']
        t_rex_status = data['t_rex_status']
        # Decode frame and put it in queue
        frame = frame.replace('data:image/png;base64,', '')
        frame += '='
        frame = frame.decode("base64")
        frame = np.fromstring(frame, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, gray_frame = cv2.threshold(gray_frame,127,255,cv2.THRESH_BINARY)
        resized_frame = cv2.resize(gray_frame, (200, 50))

        print_game_turn = False
        terminal = 0
        if game_status == GAME_OVER:
            print_game_turn = True
            GAME_TURN += 1
            reward = -100.
            terminal = 1
            # if self.debug and action == 0:
            #     scipy.misc.imsave('outfile%d.png' % GAME_TURN, gray_frame)

        # Skipping frames
        if t_rex_status == 1 and game_status != GAME_OVER:
            return None

        # if self.debug:
        #     scipy.misc.imsave('outfile%d-%s-%d.png' % (NUM_FRAME, self.dispatch_action, t_rex_status), gray_frame)

        #action = self.decode_action(action)
        NUM_FRAME += 1
        # cv2.imshow('frame', gray_frame)

        reward = 0.
        # if game_status == GAME_PLAYING and action == 1:
        #     reward = -2.

        self.frame_queue.append(resized_frame)
        self.raw_frame_queue.append(gray_frame)

        if len(self.frame_queue) == 4:
            self.prev_state = self.curr_state
            self.curr_state = self.combine_frames(self.frame_queue)

            if not self.prev_state is None:
                if len(TRAIN_DATA_QUEUE) > REPLAY_SIZE:
                    if TRAIN_DATA_QUEUE[0][-1] != 1:
                        TRAIN_DATA_QUEUE.pop(0)
                    else:
                        first = TRAIN_DATA_QUEUE.pop(0)
                        ep = random.random()
                        if ep < 0.5:
                            TRAIN_DATA_QUEUE.append(first)
                            
                
                # if self.dispatch_action == 'JUMP':
                #     scipy.misc.imsave('prev%d.png' % NUM_FRAME, self.frame_queue[2])
                #     scipy.misc.imsave('current%d.png' % NUM_FRAME, self.frame_queue[3])

                TRAIN_DATA_QUEUE.append([
                        self.prev_state, 
                        self.decode_action(self.dispatch_action), 
                        reward, 
                        self.curr_state, 
                        terminal])

            self.raw_frame_queue.pop(0)
            self.frame_queue.pop(0)

        if not self.curr_state is None:
            if game_status == GAME_PLAYING:
                ext_curr_state = self.curr_state[np.newaxis,:,:,:]
                qvalue, next_action = T_REX.take_a_action(ext_curr_state)
                next_action = int(next_action)
                self.dispatch_action = self.num_to_action(next_action)
                #print('qvalue:', qvalue.tolist(), 'action:', next_action)
                next_action = self.encode_action(next_action)
            else:
                next_action = "13"
                self.dispatch_action = 'JUMP'
                self.prev_state = None
                self.curr_state = None
                self.frame_queue = []
                self.raw_frame_queue = []

            if GAME_TURN > 0 and GAME_TURN % 5 == 0 and print_game_turn:
                print("Game turn:%d" % GAME_TURN)

            if self.emit_action:
                self.emit('recv_action', {"action": next_action})


    def recv_connect(self):
        self.dispatch_action = 'JUMP'
        self.emit('recv_action', {"action": "38"})


def socketio_service(request):
    socketio_manage(request.environ,
                    {'/bot': BotNamespace},
                    request=request)

    return Response('')

if __name__ == '__main__':
    try:
        T_REX = TRexGaming(TRAIN_DATA_QUEUE, using_cuda=False)
        T_REX.learning()
        #print('dist nhau')
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
