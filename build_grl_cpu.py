import subprocess
p = subprocess.Popen(["python", "/grl/qt-build/build.py"], cwd="/grl/qt-build")
p.wait()
