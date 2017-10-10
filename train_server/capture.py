from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response


def recv_game_frame(request):
    frame_data = request.POST['frame']
    f = open("/tmp/t-rex.png", "wb")
    f.write(frame_data.decode("base64"))
    f.close()
    return Response('Frame %(frame)s' % request.POST)


if __name__ == '__main__':
    config = Configurator()
    config.add_route('recv_frame', '/recv/')
    config.add_view(recv_game_frame, route_name='recv_frame', request_method='POST')
    app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 1234, app)
    server.serve_forever()
