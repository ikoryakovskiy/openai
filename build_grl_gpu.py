import subprocess
p = subprocess.Popen(["python3", "/grl/qt-build/cfg/build.py"], cwd="/grl/qt-build")
p.wait()
