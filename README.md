## T-Rex Deep Q-Learning
This repo was written based on the report http://cs229.stanford.edu/proj2016/report/KeZhaoWei-AIForChromeOfflineDinosaurGame-report.pdf but including some changes.
We use websocket for capturing frames to learn. We resize frames to 50x200 size and discard jumping frames (because t-rex cannot control when jumping).

## T-Rex-runner
Code base of t-rex-runner was taken from https://github.com/wayou/t-rex-runner. Some event listeners added for capturing game frames on this repo.

## Requirements
 - tensorflow 
 - opencv
 - gevent == 1.0.2
 - gevent-socketio ( https://github.com/abourget/gevent-socketio )

## Run
 - For learning:
 	```bash
 	python capture --mode=learn
 	```
 - For playing:
 	```bash
 	python capture.py --mode=play --checkpoint=path_to_checkpoint
 	```

## Trained network
 We will upload trained network soon.
